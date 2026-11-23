//! Parsing & DOM (layer 3): winnow HTML5 parser + arena DOM.
//!
//! The DOM lives in a slotmap arena. Nodes are addressed by `NodeId`
//! handles, never `Rc<RefCell<T>>`, so there are no reference cycles and
//! the JS layer can hold a `NodeId` across a re-layout without keeping
//! the node itself alive. The winnow HTML5 state-machine parser lands
//! in the html round; this file defines the arena it will fill.

use slotmap::{SlotMap, new_key_type};

new_key_type! {
    /// Handle to a node in the [`Dom`] arena.
    pub struct NodeId;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeKind {
    Document,
    Element { tag: String },
    Text(String),
}

/// A node in the arena. Children are `NodeId`s; the parent link makes
/// upward walks (style matching, event bubbling) cheap.
#[derive(Debug)]
pub struct Node {
    pub kind: NodeKind,
    pub parent: Option<NodeId>,
    pub children: Vec<NodeId>,
}

/// The arena-allocated DOM tree for one document.
#[derive(Debug, Default)]
pub struct Dom {
    arena: SlotMap<NodeId, Node>,
    root: Option<NodeId>,
}

impl Dom {
    /// Create an empty document with a single `Document` root node.
    pub fn new() -> Self {
        let mut dom = Self::default();
        let root = dom.arena.insert(Node {
            kind: NodeKind::Document,
            parent: None,
            children: Vec::new(),
        });
        dom.root = Some(root);
        dom
    }

    pub fn root(&self) -> Option<NodeId> {
        self.root
    }

    pub fn get(&self, id: NodeId) -> Option<&Node> {
        self.arena.get(id)
    }

    pub fn create_element(&mut self, tag: &str) -> NodeId {
        self.arena.insert(Node {
            kind: NodeKind::Element { tag: tag.to_string() },
            parent: None,
            children: Vec::new(),
        })
    }

    pub fn create_text(&mut self, data: &str) -> NodeId {
        self.arena.insert(Node {
            kind: NodeKind::Text(data.to_string()),
            parent: None,
            children: Vec::new(),
        })
    }

    pub fn append_child(&mut self, parent: NodeId, child: NodeId) {
        if let Some(node) = self.arena.get_mut(child) {
            node.parent = Some(parent);
        }
        if let Some(node) = self.arena.get_mut(parent) {
            node.children.push(child);
        }
    }
}
