//! Mutation-observer records (Stage 8, Recommendation 7).
//!
//! The DOM-side half of `MutationObserver`: the record type the
//! spec's "queue a mutation record" step builds, the per-observer
//! options, and the registry that matches a mutation against every
//! registered observer and queues records for the ones that match.
//!
//! This module knows nothing about JavaScript. The JS half lives in
//! `pluto-js` (`bindings::observers`): it owns the callback, calls
//! [`Dom::observe_mutation`](crate::Dom::observe_mutation) /
//! [`Dom::take_mutation_records`](crate::Dom::take_mutation_records),
//! and delivers queued records as one compound microtask per
//! observer per microtask checkpoint (see `pluto-js`'s `TabJs`).
//!
//! Delivery order: observers are notified in the order their first
//! `observe()` call happened (the spec's creation order), and the
//! records inside one observer's queue stay in mutation order. The
//! first record queued for an observer puts that observer on the
//! pending list; taking its records (delivery or `takeRecords()`)
//! clears it, so one script turn produces at most one callback per
//! observer no matter how many mutations happened.

use std::collections::HashMap;

use crate::NodeId;

/// Which kind of change a record describes (the record's `type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObserverRecordKind {
    ChildList,
    Attributes,
    CharacterData,
}

impl ObserverRecordKind {
    /// The `type` string a page sees on the record.
    pub fn type_name(self) -> &'static str {
        match self {
            ObserverRecordKind::ChildList => "childList",
            ObserverRecordKind::Attributes => "attributes",
            ObserverRecordKind::CharacterData => "characterData",
        }
    }
}

/// One spec-shaped mutation record.
///
/// `target` is the node the mutation happened on (for `childList`
/// records: the parent whose children changed; for `attributes`:
/// the element; for `characterData`: the text node). `added_nodes`
/// / `removed_nodes` are only populated for `childList` records.
/// `attribute_name` is set only for `attributes` records.
/// `old_value` is the pre-mutation value when the observer asked
/// for it (`attributeOldValue` / `characterDataOldValue`); for
/// `attributes` it is the old attribute value (`None` when the
/// attribute did not exist), for `characterData` the old text.
#[derive(Debug, Clone, PartialEq)]
pub struct ObserverMutationRecord {
    pub kind: ObserverRecordKind,
    pub target: NodeId,
    pub added_nodes: Vec<NodeId>,
    pub removed_nodes: Vec<NodeId>,
    pub previous_sibling: Option<NodeId>,
    pub next_sibling: Option<NodeId>,
    pub attribute_name: Option<String>,
    pub old_value: Option<String>,
}

impl ObserverMutationRecord {
    /// A `childList` record for `target`.
    pub fn child_list(
        target: NodeId,
        added_nodes: Vec<NodeId>,
        removed_nodes: Vec<NodeId>,
        previous_sibling: Option<NodeId>,
        next_sibling: Option<NodeId>,
    ) -> Self {
        Self {
            kind: ObserverRecordKind::ChildList,
            target,
            added_nodes,
            removed_nodes,
            previous_sibling,
            next_sibling,
            attribute_name: None,
            old_value: None,
        }
    }

    /// An `attributes` record for `target` (`old_value` is the
    /// pre-mutation value, `None` when the attribute was absent).
    pub fn attribute(target: NodeId, name: &str, old_value: Option<String>) -> Self {
        Self {
            kind: ObserverRecordKind::Attributes,
            target,
            added_nodes: Vec::new(),
            removed_nodes: Vec::new(),
            previous_sibling: None,
            next_sibling: None,
            attribute_name: Some(name.to_string()),
            old_value,
        }
    }

    /// A `characterData` record for `target` (`old_value` is the
    /// pre-mutation text).
    pub fn character_data(target: NodeId, old_value: Option<String>) -> Self {
        Self {
            kind: ObserverRecordKind::CharacterData,
            target,
            added_nodes: Vec::new(),
            removed_nodes: Vec::new(),
            previous_sibling: None,
            next_sibling: None,
            attribute_name: None,
            old_value,
        }
    }
}

/// The options passed to `observe(target, options)`.
#[derive(Debug, Clone, Default)]
pub struct MutationObserverOptions {
    /// Observe direct children being added / removed.
    pub child_list: bool,
    /// Observe attribute changes on the target.
    pub attributes: bool,
    /// Observe text-node data changes.
    pub character_data: bool,
    /// Extend the observation to the target's descendants.
    pub subtree: bool,
    /// Record the old attribute value in `attributes` records.
    pub attribute_old_value: bool,
    /// Record the old text in `characterData` records.
    pub character_data_old_value: bool,
    /// If set, only attributes in this list are observed
    /// (implies `attributes`).
    pub attribute_filter: Option<Vec<String>>,
}

impl MutationObserverOptions {
    /// True when at least one observation type is on. `observe()`
    /// throws `TypeError` when this is false (an `attributeFilter`
    /// alone counts: it implies `attributes`).
    pub fn observes_any(&self) -> bool {
        self.child_list
            || self.attributes
            || self.character_data
            || self
                .attribute_filter
                .as_ref()
                .is_some_and(|f| !f.is_empty())
    }

    /// True when `attributes` records are wanted (explicitly, or
    /// implied by a non-empty `attributeFilter`).
    pub fn wants_attributes(&self) -> bool {
        self.attributes
            || self
                .attribute_filter
                .as_ref()
                .is_some_and(|f| !f.is_empty())
    }
}

/// One observer's registration against one target node.
#[derive(Debug, Clone)]
struct Registration {
    observer_id: u64,
    target: NodeId,
    options: MutationObserverOptions,
}

/// The DOM-side mutation-observer registry.
///
/// Matching is done by the `Dom` (it owns the parent chain, so it
/// can test `subtree`); this registry only stores registrations
/// and queues. `queue` is the only way records enter: the first
/// record for an observer puts it on the pending list, in the
/// order the first record arrived.
#[derive(Debug, Clone, Default)]
pub struct MutationObserverRegistry {
    registrations: Vec<Registration>,
    queues: HashMap<u64, Vec<ObserverMutationRecord>>,
    pending: Vec<u64>,
    observer_order: Vec<u64>,
}

impl MutationObserverRegistry {
    /// Register `observer_id` on `target` (replaces an existing
    /// registration for the same observer + target pair, per spec).
    pub fn observe(&mut self, observer_id: u64, target: NodeId, options: MutationObserverOptions) {
        self.registrations
            .retain(|r| !(r.observer_id == observer_id && r.target == target));
        self.registrations.push(Registration {
            observer_id,
            target,
            options,
        });
        if !self.observer_order.contains(&observer_id) {
            self.observer_order.push(observer_id);
        }
    }

    /// All registrations, in registration order (the `Dom` matches
    /// each one against the mutated node).
    pub fn registrations(&self) -> Vec<(u64, NodeId, MutationObserverOptions)> {
        self.registrations
            .iter()
            .map(|r| (r.observer_id, r.target, r.options.clone()))
            .collect()
    }

    /// Queue `record` for `observer_id`. The observer joins the
    /// pending list only if its queue was empty — one pending entry
    /// per observer per delivery round, however many records arrive.
    pub fn queue(&mut self, observer_id: u64, record: ObserverMutationRecord) {
        let queue = self.queues.entry(observer_id).or_default();
        let was_empty = queue.is_empty();
        queue.push(record);
        if was_empty && !self.pending.contains(&observer_id) {
            self.pending.push(observer_id);
        }
    }

    /// Observers with undelivered records, in the order their
    /// first record arrived (which is observer-creation order for
    /// records queued in the same turn).
    pub fn pending_observers(&self) -> Vec<u64> {
        self.pending.clone()
    }

    /// Take the pending list, clearing it. Delivery then drains
    /// each observer's queue with [`Self::take_records`].
    pub fn take_pending_observers(&mut self) -> Vec<u64> {
        std::mem::take(&mut self.pending)
    }

    /// Drain one observer's queued records (delivery, or
    /// `takeRecords()`). Also clears its pending entry.
    pub fn take_records(&mut self, observer_id: u64) -> Vec<ObserverMutationRecord> {
        self.pending.retain(|&id| id != observer_id);
        self.queues.remove(&observer_id).unwrap_or_default()
    }

    /// Drop an observer entirely: registrations, queued records,
    /// and its pending entry. A later `observe()` re-registers it
    /// fresh (the observer's creation order is kept, so a
    /// re-observed observer still notifies in its original place).
    pub fn disconnect(&mut self, observer_id: u64) {
        self.registrations.retain(|r| r.observer_id != observer_id);
        self.queues.remove(&observer_id);
        self.pending.retain(|&id| id != observer_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options(
        child_list: bool,
        attributes: bool,
        character_data: bool,
    ) -> MutationObserverOptions {
        MutationObserverOptions {
            child_list,
            attributes,
            character_data,
            ..Default::default()
        }
    }

    #[test]
    fn first_record_pends_once_and_take_drains() {
        let mut reg = MutationObserverRegistry::default();
        let mut dom = crate::Dom::new();
        let target = dom.create_element("div");
        let added = dom.create_element("span");
        reg.observe(7, target, options(true, false, false));
        let rec = ObserverMutationRecord::child_list(target, vec![added], vec![], None, None);
        reg.queue(7, rec.clone());
        reg.queue(7, rec.clone());
        assert_eq!(reg.pending_observers(), vec![7]);
        assert_eq!(reg.take_records(7).len(), 2);
        assert!(reg.pending_observers().is_empty());
        assert!(reg.take_records(7).is_empty());
    }

    #[test]
    fn observe_replaces_registration_for_same_target() {
        let mut reg = MutationObserverRegistry::default();
        let mut dom = crate::Dom::new();
        let target = dom.create_element("div");
        reg.observe(3, target, options(true, false, false));
        reg.observe(3, target, options(false, true, false));
        let regs = reg.registrations();
        assert_eq!(regs.len(), 1);
        assert!(!regs[0].2.child_list && regs[0].2.attributes);
    }

    #[test]
    fn disconnect_discards_queue_and_registrations() {
        let mut reg = MutationObserverRegistry::default();
        let mut dom = crate::Dom::new();
        let target = dom.create_element("div");
        reg.observe(9, target, options(true, true, true));
        reg.queue(
            9,
            ObserverMutationRecord::attribute(target, "class", Some("a".into())),
        );
        assert_eq!(reg.pending_observers(), vec![9]);
        reg.disconnect(9);
        assert!(reg.pending_observers().is_empty());
        assert!(reg.take_records(9).is_empty());
        assert!(reg.registrations().is_empty());
        // Re-observing after disconnect works.
        reg.observe(9, target, options(true, false, false));
        assert_eq!(reg.registrations().len(), 1);
    }

    #[test]
    fn options_observes_any_and_filter_implies_attributes() {
        assert!(!MutationObserverOptions::default().observes_any());
        assert!(options(true, false, false).observes_any());
        let filtered = MutationObserverOptions {
            attribute_filter: Some(vec!["id".into()]),
            ..Default::default()
        };
        assert!(filtered.observes_any());
        assert!(filtered.wants_attributes());
        assert!(!MutationObserverOptions::default().wants_attributes());
    }
}
