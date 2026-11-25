//! The host side of the binding boundary: what `pluto-js` needs from the
//! DOM and the layout actor (Stage 7, Recommendation 5).
//!
//! Wave 2 implements [`DomHost`] against `pluto-html`'s `Dom` plus a
//! channel to the layout actor. Nothing here mentions the JS engine:
//! a binding holds a [`DomKey`], validates it on every crossing, and
//! never holds a pointer or a GC root into the arena.
//!
//! # Safety model (from the Stage 7 report, §(c))
//!
//! * A JS wrapper holds only a [`DomKey`] — a generational slotmap key
//!   into the arena. The arena owns the node; the key is just an index
//!   plus a generation.
//! * Every crossing validates the key against the live `Dom`. A removed
//!   node (or a reused slot from a later node) fails validation:
//!   getters return `None` (the binding surfaces `null`), setters are
//!   no-ops or throw `TypeError` per WebIDL, never use-after-free.
//! * Wrapper identity (`document.body === document.body`) is a
//!   per-realm cache keyed by `DomKey`, built in wave 2 on top of this
//!   trait — it is not needed to validate a key.

use pluto_html::NodeId;
use pluto_html::observers::{MutationObserverOptions, ObserverMutationRecord};

/// A handle to a DOM node, as held by JS wrappers. This is
/// `pluto-html`'s generational slotmap key: the index identifies the
/// arena slot, the generation identifies which occupant of that slot.
/// A key for a removed node never validates again, even if the slot is
/// later reused by a different node.
pub type DomKey = NodeId;

/// A layout box, as returned by a forced synchronous layout flush.
/// All values are CSS pixels in the document's coordinate space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Geometry {
    /// X offset of the border box relative to the document origin.
    pub x: f64,
    /// Y offset of the border box relative to the document origin.
    pub y: f64,
    /// Border-box width.
    pub width: f64,
    /// Border-box height.
    pub height: f64,
}

/// Why a host tree operation failed, so the binding can raise the
/// right DOMException name (WebIDL: tree surgery throws
/// `HierarchyRequestError` / `NotFoundError`, never a bare bool).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DomHostError {
    /// The key does not address a live node (stale / wrong document).
    StaleKey,
    /// The operation would break the tree (node into itself or one of
    /// its descendants, a Document node as a child, …).
    Hierarchy,
    /// A named child (`old`, `sibling`) is not a child of `parent`.
    NotFound,
    /// The document is in a state that forbids the call (e.g.
    /// `document.write` after the document was closed with no parser).
    InvalidState,
}

/// What the JS engine needs from the host document. Implemented in wave
/// 2 by the per-tab worker (which owns the `Dom` and talks to the
/// layout actor). All methods are synchronous and run on the tab's
/// worker thread — the only thread that ever touches the `Dom`.
///
/// TODO(M5): when the worker moves across IPC, this trait's implementor
/// becomes a channel client; the trait itself does not change.
///
/// # Wave-2a extensions
///
/// The methods below the wave-1 block are additive (new methods only;
/// no wave-1 signature changed). They carry default implementations
/// that report "not present" so a minimal host still compiles; the
/// production host (per-tab worker over `pluto-html`'s `Dom`, with
/// `pluto-layout::selector_matches` for queries) must override them —
/// a default that returns `None` / empty is a missing feature, not a
/// correct answer, and the bindings document each one. The test host
/// in `pluto-js/src/bindings/test_host.rs` overrides all of them.
pub trait DomHost {
    /// True if `key` still addresses a live node in this document.
    /// Called on every crossing, before any other method.
    fn validate(&self, key: DomKey) -> bool;

    /// The document element (`<html>`), if the document has one.
    fn document_element(&self) -> Option<DomKey>;

    /// The `<body>` element, if the document has one.
    fn body(&self) -> Option<DomKey>;

    /// First element with this `id` in document order, or `None`.
    fn get_element_by_id(&self, id: &str) -> Option<DomKey>;

    /// The `textContent` of a node: for an element, the concatenation of
    /// its descendant text; for a text node, its data. `None` if the key
    /// is stale or the node has no text.
    fn text_content(&self, key: DomKey) -> Option<String>;

    /// Set `textContent` (replaces children with a single text node).
    /// Returns false if the key is stale, in which case the binding
    /// surfaces a `TypeError` or a no-op per WebIDL.
    fn set_text_content(&mut self, key: DomKey, text: &str) -> bool;

    /// Set a text node's data (`CharacterData.data`). Returns false
    /// if the key is stale or not a text node. Stage 8 Rec 7: this
    /// is the JS-visible character-data mutation (it queues a
    /// `characterData` record with the old text).
    fn set_character_data(&mut self, _key: DomKey, _data: &str) -> bool {
        false
    }

    /// A content attribute value, or `None` if absent / key stale.
    fn get_attribute(&self, key: DomKey, name: &str) -> Option<String>;

    /// Set a content attribute. Returns false if the key is stale.
    /// Setting an attribute marks the node's dirty bits (style and
    /// layout, self + subtree) — the write path is lazy, no layout
    /// happens here.
    fn set_attribute(&mut self, key: DomKey, name: &str, value: &str) -> bool;

    /// Mark the subtree rooted at `key` as needing restyle and layout
    /// (sets the Double Dirty Bit flags, self + subtree). This is the
    /// write path: it records the dirt and returns immediately. The
    /// actual restyle/layout happens at the next rendering opportunity.
    fn request_restlye_and_layout(&mut self, key: DomKey);

    /// Geometry of a node, forcing a synchronous layout flush first if
    /// the layout tree is dirty. This is the ONLY sanctioned synchronous
    /// cross-thread call in the design (worker thread blocks on the
    /// layout actor), and it is bounded: the host flushes only the dirty
    /// subtree needed for the query where it can, else the whole
    /// document (wave 2 documents which one it ships). Returns `None`
    /// if the key is stale.
    ///
    /// The host counts forced flushes; see
    /// [`DomHost::forced_flush_count`]. A read/write loop that flushes
    /// once per iteration is layout thrash by design — visible here,
    /// not hidden.
    fn geometry(&mut self, key: DomKey) -> Option<Geometry>;

    /// How many forced synchronous layout flushes this document has
    /// performed. A debug counter so layout thrash is measurable in
    /// tests (wave 2 wires it to the layout actor).
    fn forced_flush_count(&self) -> u64;

    // --- Wave-2a extensions (additive; defaults report "not present") ---

    /// The `Document` node itself (the root of the arena), if any.
    /// The `document` wrapper holds this key.
    fn document_node(&self) -> Option<DomKey> {
        None
    }

    /// The `<head>` element, if the document has one.
    fn head(&self) -> Option<DomKey> {
        None
    }

    /// Parent of `key`, or `None` for the document root / stale key.
    fn parent(&self, _key: DomKey) -> Option<DomKey> {
        None
    }

    /// Children of `key` in document order (empty for stale keys).
    fn children(&self, _key: DomKey) -> Vec<DomKey> {
        Vec::new()
    }

    /// Element tag name as stored (lowercase from the parser), or
    /// `None` for non-elements / stale keys. The binding uppercases
    /// it for `tagName` per HTML.
    fn tag_name(&self, _key: DomKey) -> Option<String> {
        None
    }

    /// DOM `nodeType`: 1 element, 3 text, 8 comment, 9 document,
    /// 10 doctype. `None` for stale keys.
    fn node_type(&self, _key: DomKey) -> Option<u16> {
        None
    }

    /// DOM `nodeName`: uppercase tag for elements, `#text` /
    /// `#comment` / `#document`, doctype name. `None` for stale keys.
    fn node_name(&self, _key: DomKey) -> Option<String> {
        None
    }

    /// The document title (text of the first `<title>` in `<head>`),
    /// or `None` if there is none.
    fn title(&self) -> Option<String> {
        None
    }

    /// Set the document title (creates/updates the `<title>` element).
    /// Returns false if the document has no `<head>` / key is stale.
    fn set_title(&mut self, _title: &str) -> bool {
        false
    }

    /// Create a detached element node. `None` only on allocation
    /// failure (never in practice); the tag is stored as given.
    fn create_element(&mut self, _tag: &str) -> Option<DomKey> {
        None
    }

    /// Create a detached text node.
    fn create_text_node(&mut self, _data: &str) -> Option<DomKey> {
        None
    }

    /// Append `child` under `parent` (detaching it first). The host
    /// enforces the hierarchy rules and reports why it refused, so
    /// the binding can raise the right DOMException.
    fn append_child(&mut self, _parent: DomKey, _child: DomKey) -> Result<(), DomHostError> {
        Err(DomHostError::StaleKey)
    }

    /// Insert `child` under `parent` before `sibling` (`None` =
    /// append). `sibling` must be a child of `parent`.
    fn insert_before(
        &mut self,
        _parent: DomKey,
        _child: DomKey,
        _sibling: Option<DomKey>,
    ) -> Result<(), DomHostError> {
        Err(DomHostError::StaleKey)
    }

    /// Detach `child` from `parent`. `child` must be a child of
    /// `parent` (else `NotFound`).
    fn remove_child(&mut self, _parent: DomKey, _child: DomKey) -> Result<(), DomHostError> {
        Err(DomHostError::StaleKey)
    }

    /// Replace `old` with `new` under `parent`, keeping position.
    fn replace_child(
        &mut self,
        _parent: DomKey,
        _old: DomKey,
        _new: DomKey,
    ) -> Result<(), DomHostError> {
        Err(DomHostError::StaleKey)
    }

    /// Remove a content attribute. Returns false if the key is stale
    /// (removing an absent attribute is a silent success).
    fn remove_attribute(&mut self, _key: DomKey, _name: &str) -> bool {
        false
    }

    /// True if the element has a content attribute of this name.
    fn has_attribute(&self, _key: DomKey, _name: &str) -> bool {
        false
    }

    // --- Stage 13 (M5 forms) extensions (additive; defaults report
    // "not a control" so hosts without form state keep compiling) ---

    /// Whether `key` is a form control (`input` / `textarea` /
    /// `select`).
    fn is_form_control(&self, _key: DomKey) -> bool {
        false
    }

    /// The `<input>` `type` IDL value (lowercased, `"text"` default).
    fn input_type(&self, _key: DomKey) -> Option<String> {
        None
    }

    /// Whether the control is disabled (own attribute or a disabled
    /// `<fieldset>` ancestor).
    fn is_disabled(&self, _key: DomKey) -> bool {
        false
    }

    /// Live `.value` of a control, or `None` for non-controls.
    fn form_value(&self, _key: DomKey) -> Option<String> {
        None
    }

    /// Default (parse-time) value of a control.
    fn form_default_value(&self, _key: DomKey) -> Option<String> {
        None
    }

    /// Whether the live value was set by typing / the IDL setter.
    fn form_value_dirty(&self, _key: DomKey) -> bool {
        false
    }

    /// Set the live value (IDL `.value` setter): marks it dirty.
    fn set_form_value(&mut self, _key: DomKey, _value: &str) -> bool {
        false
    }

    /// Live checkedness of a checkbox / radio.
    fn form_checked(&self, _key: DomKey) -> Option<bool> {
        None
    }

    /// Set the live checkedness (IDL `.checked` setter).
    fn set_form_checked(&mut self, _key: DomKey, _checked: bool) -> bool {
        false
    }

    /// Live selected index of a `<select>`.
    fn form_selected_index(&self, _key: DomKey) -> Option<i64> {
        None
    }

    /// Set the selected index of a `<select>`.
    fn set_form_selected_index(&mut self, _key: DomKey, _index: i64) -> bool {
        false
    }

    /// A `<select>`'s `<option>` keys in tree order.
    fn form_options(&self, _key: DomKey) -> Vec<DomKey> {
        Vec::new()
    }

    /// The `<form>` that owns a control, or `None`.
    fn form_owner(&self, _key: DomKey) -> Option<DomKey> {
        None
    }

    /// All controls owned by a form, in document order.
    fn form_controls(&self, _key: DomKey) -> Vec<DomKey> {
        Vec::new()
    }

    /// All `<form>` elements in the document, in document order.
    fn document_forms(&self) -> Vec<DomKey> {
        Vec::new()
    }

    /// The submission entry list of a form: successful controls
    /// only, `(name, value)` in document order. `submitter` is the
    /// button that submitted, if any.
    fn form_entries(&self, _key: DomKey, _submitter: Option<DomKey>) -> Vec<(String, String)> {
        Vec::new()
    }

    /// Reset a form's controls to their seeded defaults.
    fn reset_form(&mut self, _key: DomKey) -> bool {
        false
    }

    /// Caret offset (chars) of a text control.
    fn form_caret(&self, _key: DomKey) -> Option<usize> {
        None
    }

    /// Selection range `[start, end)` of a text control.
    fn form_selection(&self, _key: DomKey) -> Option<(usize, usize)> {
        None
    }

    /// Set the caret / selection of a text control.
    /// Insert `text` at the caret (replacing the selection),
    /// honouring `maxlength` (Stage 13 Rec 4). Returns the new
    /// caret position, or `None` when the node is not an editable
    /// text control.
    fn insert_form_text(&mut self, _key: DomKey, _text: &str) -> Option<usize> {
        None
    }

    /// Delete one char before (`backspace`) or after the caret
    /// (or the whole selection) (Stage 13 Rec 4). Returns the new
    /// caret position, or `None` when the node is not editable.
    fn delete_form_text(&mut self, _key: DomKey, _backspace: bool) -> Option<usize> {
        None
    }

    fn set_form_selection(&mut self, _key: DomKey, _start: usize, _end: usize) -> bool {
        false
    }

    /// The custom validity message set by `setCustomValidity`
    /// (empty = no custom error).
    fn custom_validity(&self, _key: DomKey) -> Option<String> {
        None
    }

    /// Set the custom validity message (empty clears it).
    fn set_custom_validity(&mut self, _key: DomKey, _message: &str) -> bool {
        false
    }

    /// The currently focused element (`document.activeElement`),
    /// or `None` when nothing is focused.
    fn active_element(&self) -> Option<DomKey> {
        None
    }

    /// Focus `key` (the `focus()` IDL method): returns false when
    /// the node cannot take focus. The renderer moves its own
    /// focus state too; this only records the host-side focus the
    /// bindings read back.
    /// Take a focus change queued by `focus_element` /
    /// `blur_element` for the renderer to apply (Stage 13 Rec 3):
    /// the host cannot move the tab's focus itself — the tab
    /// thread owns `state.focused` — so JS-driven focus is a
    /// pending request the tab thread drains after each JS run
    /// (like `take_pending_navigation`). `None` = no pending
    /// change; `Some(Some(key))` = focus that node;
    /// `Some(None)` = blur.
    fn take_pending_focus(&mut self) -> Option<Option<DomKey>> {
        None
    }

    fn focus_element(&mut self, _key: DomKey) -> bool {
        false
    }

    /// Blur the focused element (the `blur()` IDL method).
    fn blur_element(&mut self) {}

    /// `document.cookie` for the document origin: non-HttpOnly
    /// cookies as `name=value; ...`, or `None` when the host has
    /// no jar.
    fn document_cookie(&self) -> Option<String> {
        None
    }

    /// Set one cookie from `document.cookie = "..."` (a single
    /// `name=value` pair plus attributes, `store_set_cookie`
    /// semantics). Returns false when the host has no jar.
    fn set_document_cookie(&mut self, _cookie: &str) -> bool {
        false
    }

    /// Queue a form submission for the renderer to perform
    /// (`form.submit()` / `requestSubmit()` after validation and
    /// the `submit` event): the tab thread drains the queue and
    /// navigates through the loader. Returns false when the host
    /// cannot submit (no renderer attached).
    fn request_form_submit(&mut self, _form: DomKey, _submitter: Option<DomKey>) -> bool {
        false
    }

    /// Drain queued form submissions, oldest first.
    fn take_pending_form_submits(&mut self) -> Vec<(DomKey, Option<DomKey>)> {
        Vec::new()
    }

    /// First element in `scope`'s subtree (including `scope` itself
    /// if it is an element) matching `selector`, in document order.
    /// The production host implements this with the Stage-6 selector
    /// matcher (`pluto-layout::selector_matches` over the parsed
    /// `pluto-css::Selector`); the binding never matches selectors
    /// itself. An invalid selector returns `None` (the binding
    /// surfaces `null` for `querySelector` — M4 departure: the spec
    /// throws `SyntaxError` for an invalid selector; see the binding).
    fn query_selector(&self, _scope: DomKey, _selector: &str) -> Option<DomKey> {
        None
    }

    /// All elements in `scope`'s subtree matching `selector`, in
    /// document order. Same matcher note as [`DomHost::query_selector`].
    fn query_selector_all(&self, _scope: DomKey, _selector: &str) -> Vec<DomKey> {
        Vec::new()
    }

    /// Serialised `innerHTML` of `key` (its children as markup).
    /// `None` for stale keys.
    fn inner_html(&self, _key: DomKey) -> Option<String> {
        None
    }

    /// Parse `html` as a fragment in the context of `key` and replace
    /// its children with the fragment's nodes. `<script>` elements the
    /// fragment creates are inert (they never execute). Returns false
    /// if the key is stale. The fragment parser is the parser-seams
    /// agent's `parse_fragment` in `pluto-html`; this trait only
    /// carries the call across the boundary.
    fn set_inner_html(&mut self, _key: DomKey, _html: &str) -> bool {
        false
    }

    /// `document.write(text)`: insert markup at the parser's insertion
    /// point (parser agent's stack), or — with no active parser — the
    /// implicit-`open()` path. Returns false when the document state
    /// forbids the write; the binding raises `InvalidStateError`.
    fn document_write(&mut self, _text: &str) -> bool {
        false
    }

    /// `document.open()`: replace the document in place. Returns false
    /// when called from a parser-inserted script (ignored per spec).
    fn document_open(&mut self) -> bool {
        false
    }

    /// `document.close()`: finish the document (emit EOF). Returns
    /// false when there is nothing to close.
    fn document_close(&mut self) -> bool {
        false
    }

    /// The raw `style` attribute text of `key` (the CSSOM
    /// `cssText`), or `None` for stale keys / no `style` attribute.
    fn style_text(&self, _key: DomKey) -> Option<String> {
        None
    }

    /// Set the raw `style` attribute text of `key`. Returns false if
    /// the key is stale.
    fn set_style_text(&mut self, _key: DomKey, _css: &str) -> bool {
        false
    }

    /// A script/callback error surfaced to the host (the integration
    /// wave wires this to `window.onerror` reporting / the shell log).
    /// Default: ignore — `TabJs::script_errors` already records it.
    fn report_error(&self, _message: &str) {}

    // --- Stage 12 Rec 7: binding-batch host hooks (defaults are
    // "not present"; the production host overrides them, the test
    // host implements them — a default here is a missing feature,
    // not a correct answer) ---

    /// The document's URL (the `window.location` base), if the host
    /// tracks one. `None` means the realm cannot know the URL and
    /// `location` reads as `about:blank`.
    fn document_url(&self) -> Option<String> {
        None
    }

    /// Ask the shell to navigate to `url` (`replace` = `location.replace`
    /// semantics, no new history entry). The realm never navigates
    /// itself: the shell performs the navigation and the next
    /// document's `document_url` reflects it. Default: ignore.
    fn request_navigation(&mut self, _url: &str, _replace: bool) {}

    /// Attribute names of `key`, in source order (empty for stale
    /// keys). Backs `dataset` (which needs the `data-*` subset) and
    /// `cloneNode` (which copies attributes).
    fn attribute_names(&self, _key: DomKey) -> Vec<String> {
        Vec::new()
    }

    /// Deep- (`deep` = true) or shallow-clone `key` into a detached
    /// node: same tag / text / attributes, children copied only
    /// when `deep`. `None` for stale keys. The binding wraps the
    /// result; the clone is not inserted anywhere.
    fn clone_node(&mut self, _key: DomKey, _deep: bool) -> Option<DomKey> {
        None
    }

    /// Ask the shell to fetch + decode an image at `url` (resolved
    /// against the document base `base_url`), for an `Image()`
    /// object or an `<img>` whose `src` was set from JS. The shell
    /// routes this through the Stage 11 image store (same gates as
    /// the preload scanner: mixed content, blocker, HTTP cache)
    /// and reports back via `TabJs::notify_image_loaded` /
    /// `notify_image_failed`, which fire `load` / `error` on the
    /// element. Default: ignore (the image never loads).
    fn request_image(&mut self, _url: &str, _base_url: &str) {}

    // --- Stage 8 Rec 7: MutationObserver (DOM-side registry) ---

    /// Register a mutation observer on `target`. Returns false when
    /// the target is not a live node. Default: no registry — the
    /// observer never fires (the production and test hosts override
    /// this by delegating to `pluto-html`'s `Dom`).
    fn observe_mutation(
        &mut self,
        _observer_id: u64,
        _target: DomKey,
        _options: MutationObserverOptions,
    ) -> bool {
        false
    }

    /// Drop a mutation observer: registrations and queued records
    /// are discarded.
    fn disconnect_mutation_observer(&mut self, _observer_id: u64) {}

    /// Drain one mutation observer's queued records.
    fn take_mutation_records(&mut self, _observer_id: u64) -> Vec<ObserverMutationRecord> {
        Vec::new()
    }

    /// Mutation observers with undelivered records, in
    /// first-record order (peek — does not drain).
    fn pending_mutation_observers(&self) -> Vec<u64> {
        Vec::new()
    }

    /// Mutation observers with undelivered records, in
    /// first-record order. Drained by the loop's microtask
    /// checkpoint before delivery.
    fn take_pending_mutation_observers(&mut self) -> Vec<u64> {
        Vec::new()
    }

    // --- Stage 8 Rec 8: ResizeObserver (layout sizes) ---

    /// Content-box size of `key` in CSS px, or `None` when the key
    /// is stale / unrendered. Default: the border-box size from
    /// [`DomHost::geometry`] (no padding data at this layer — the
    /// browser's worker step supplies real content sizes from the
    /// layout tree instead).
    fn content_size(&mut self, key: DomKey) -> Option<(f64, f64)> {
        self.geometry(key).map(|g| (g.width, g.height))
    }

    /// DOM depth of `key` (document root = 0), or `None` when the
    /// key is stale. The resize gather orders entries
    /// shallowest-first; `None` is treated as depth 0.
    fn depth(&self, _key: DomKey) -> Option<usize> {
        None
    }
}
