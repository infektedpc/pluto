//! Engine probe (Stage 7, Recommendation 2): the regression gate for
//! engine upgrades.
//!
//! [`probe_engine`] runs a feature checklist plus a small Test262-style
//! subset against the pinned engine and returns one pass/fail row per
//! item. This resolves the report's ES2020-vs-ES2023 discrepancy WITH
//! DATA — the verdict is recorded in `pluto-js/README.md` from the
//! output of this probe, not from the engine's marketing copy.
//!
//! The probe runs under `cargo test -p pluto-js` (see the tests at the
//! bottom of this module). Do not delete it: the next engine upgrade
//! re-runs it, and any newly-failing row blocks the upgrade.

use crate::engine::{EvalKind, JsEngine, JsValue};
use crate::realm::{RealmConfig, TabRealm};

/// One probe row: a feature (or a Test262-style behaviour) and whether
/// the pinned engine implements it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbeItem {
    /// Feature name, e.g. `"modules"` or `"test262: classes"`.
    pub name: &'static str,
    /// True if the engine behaves as the feature requires.
    pub passed: bool,
    /// What happened: the observed value, or the engine's error.
    pub detail: String,
}

/// Run the whole checklist against a fresh realm of the pinned engine.
/// Each item is independent: a failure is recorded, never fatal to the
/// rest of the probe.
pub fn probe_engine() -> Vec<ProbeItem> {
    let mut realm = match TabRealm::new(&RealmConfig::default()) {
        Ok(realm) => realm,
        Err(e) => {
            return vec![ProbeItem {
                name: "realm-creation",
                passed: false,
                detail: format!("could not create a realm: {e}"),
            }];
        }
    };
    let mut items = Vec::new();

    // --- Feature checklist (report §(a): ES2020 vs ES2023) ---

    // Modules: declare + evaluate, with an export and a global side
    // effect (the export itself is checked via the global, since the
    // module's completion value is `undefined`).
    match realm.eval_module(
        "probe.mjs",
        "export const answer = 42; globalThis.__probe_answer = 42;",
    ) {
        Ok(_) => {
            let ok = realm
                .eval(EvalKind::Classic, "globalThis.__probe_answer === 42", "probe.js")
                .map(|v| v == JsValue::Bool(true))
                .unwrap_or(false);
            items.push(ProbeItem {
                name: "modules",
                passed: ok,
                detail: if ok {
                    "module evaluated, export + global side effect visible".to_string()
                } else {
                    "module evaluated but its side effect was not visible".to_string()
                },
            });
        }
        Err(e) => items.push(ProbeItem {
            name: "modules",
            passed: false,
            detail: format!("module evaluation failed: {e}"),
        }),
    }

    check_async_value(
        &mut realm,
        &mut items,
        "top-level-await",
        "globalThis.__probe_v = await Promise.resolve(42);",
        JsValue::Number(42.0),
    );

    check_async_value(
        &mut realm,
        &mut items,
        "async-generators",
        "await (async () => { \
           async function* gen() { yield 1; yield 2; yield 3; } \
           const out = []; \
           for await (const v of gen()) out.push(v); \
           globalThis.__probe_v = out.join(','); \
         })()",
        JsValue::String("1,2,3".to_string()),
    );

    check_bool(
        &mut realm,
        &mut items,
        "bigint",
        "10n ** 20n === 100000000000000000000n",
    );

    check_bool(
        &mut realm,
        &mut items,
        "proxy",
        "(() => { \
           const p = new Proxy({ x: 1 }, { get(t, k) { return k === 'x' ? 42 : t[k]; } }); \
           return p.x === 42; \
         })()",
    );

    // Temporal: expected to be absent (QuickJS-NG has no Temporal). A
    // fail here is data, not a bug: it bounds the ES-version verdict.
    check_bool(
        &mut realm,
        &mut items,
        "temporal",
        "typeof Temporal !== 'undefined'",
    );

    // Array.fromAsync (ES2023): typeof check first, then a real run.
    match realm.eval(
        EvalKind::Classic,
        "typeof Array.fromAsync === 'function'",
        "probe.js",
    ) {
        Ok(JsValue::Bool(true)) => check_async_value(
            &mut realm,
            &mut items,
            "array-fromAsync",
            "globalThis.__probe_v = (await Array.fromAsync([Promise.resolve(1), Promise.resolve(2)])).join(',');",
            JsValue::String("1,2".to_string()),
        ),
        Ok(other) => items.push(ProbeItem {
            name: "array-fromAsync",
            passed: false,
            detail: format!("typeof Array.fromAsync is not 'function' (got {other:?})"),
        }),
        Err(e) => items.push(ProbeItem {
            name: "array-fromAsync",
            passed: false,
            detail: format!("typeof check failed: {e}"),
        }),
    }

    // Decorators (ES2022, standard form): a decorator that returns the
    // method unchanged.
    check_bool(
        &mut realm,
        &mut items,
        "decorators",
        "(() => { \
           function dec(value, context) { return value; } \
           class C { @dec m() { return 1; } } \
           return new C().m() === 1; \
         })()",
    );

    // --- Test262-style subset (behaviour, not just typeof) ---

    // Promise ordering: microtasks run after the sync task, in order.
    let _ = realm.eval(
        EvalKind::Classic,
        "globalThis.__probe_order = []; \
         Promise.resolve().then(() => __probe_order.push('micro')); \
         __probe_order.push('sync');",
        "probe.js",
    );
    let _ = realm.drain_jobs();
    match realm.eval(
        EvalKind::Classic,
        "__probe_order.join(',')",
        "probe.js",
    ) {
        Ok(JsValue::String(s)) => items.push(ProbeItem {
            name: "test262: promise-ordering",
            passed: s == "sync,micro",
            detail: format!("order was {s:?}"),
        }),
        Ok(other) => items.push(ProbeItem {
            name: "test262: promise-ordering",
            passed: false,
            detail: format!("unexpected value {other:?}"),
        }),
        Err(e) => items.push(ProbeItem {
            name: "test262: promise-ordering",
            passed: false,
            detail: format!("failed: {e}"),
        }),
    }

    check_async_value(
        &mut realm,
        &mut items,
        "test262: async-await",
        "await (async () => { const v = await Promise.resolve(41); globalThis.__probe_v = v + 1; })()",
        JsValue::Number(42.0),
    );

    check_bool(
        &mut realm,
        &mut items,
        "test262: classes",
        "(() => { \
           class A { greet() { return 'hi'; } } \
           class B extends A { greet() { return super.greet() + '!'; } } \
           return new B().greet() === 'hi!'; \
         })()",
    );

    check_bool(
        &mut realm,
        &mut items,
        "test262: generators",
        "(() => { \
           function* g() { yield 1; yield 2; } \
           const it = g(); \
           return it.next().value === 1 && it.next().value === 2 && it.next().done === true; \
         })()",
    );

    check_bool(
        &mut realm,
        &mut items,
        "test262: destructuring",
        "(() => { \
           const [a, , c = 3] = [1, 2]; \
           const { x: { y } = {} } = {}; \
           const { p = 9, ...rest } = { p: 1, q: 2 }; \
           return a === 1 && c === 3 && y === undefined && p === 1 && rest.q === 2; \
         })()",
    );

    check_bool(
        &mut realm,
        &mut items,
        "test262: optional-chaining",
        "(() => { \
           const o = { a: { b: 2 } }; \
           return o?.a?.b === 2 && null?.a === undefined && o?.missing?.deep === undefined; \
         })()",
    );

    check_bool(
        &mut realm,
        &mut items,
        "test262: nullish-coalescing",
        "(null ?? 'x') === 'x' && (0 ?? 'x') === 0 && ('' ?? 'x') === '' \
         && (false ?? 'x') === false && (undefined ?? 'x') === 'x'",
    );

    check_async_value(
        &mut realm,
        &mut items,
        "test262: promise-all",
        "globalThis.__probe_v = (await Promise.all([Promise.resolve(1), Promise.resolve(2)])).join(',');",
        JsValue::String("1,2".to_string()),
    );

    items
}

/// Evaluate a classic script that must produce `true`.
fn check_bool(realm: &mut TabRealm, items: &mut Vec<ProbeItem>, name: &'static str, source: &str) {
    match realm.eval(EvalKind::Classic, source, "probe.js") {
        Ok(JsValue::Bool(true)) => items.push(ProbeItem {
            name,
            passed: true,
            detail: "evaluated to true".to_string(),
        }),
        Ok(other) => items.push(ProbeItem {
            name,
            passed: false,
            detail: format!("evaluated to {other:?}, expected true"),
        }),
        Err(e) => items.push(ProbeItem {
            name,
            passed: false,
            detail: format!("failed: {e}"),
        }),
    }
}

/// Evaluate a top-level-await script that must store its result in
/// `globalThis.__probe_v`, then read the global back with a classic
/// eval and compare.
///
/// Rationale (verified empirically against QuickJS-NG 0.15.1):
/// `Ctx::eval_promise` (JS_Eval + JS_EVAL_FLAG_ASYNC) fulfils with
/// `{ value: <completion> }`, not with the completion itself, so the
/// eval promise's fulfilment value cannot be read as the script's
/// value; capturing into a global and reading it back is the honest
/// way to observe async results. The global is cleared first so a
/// script that throws before assigning cannot pass on a stale value.
fn check_async_value(
    realm: &mut TabRealm,
    items: &mut Vec<ProbeItem>,
    name: &'static str,
    source: &str,
    expected: JsValue,
) {
    let _ = realm.eval(
        EvalKind::Classic,
        "globalThis.__probe_v = undefined;",
        "probe.js",
    );
    match realm.eval(EvalKind::ClassicAwait, source, "probe.js") {
        Ok(_) => match realm.eval(EvalKind::Classic, "globalThis.__probe_v", "probe.js") {
            Ok(value) => items.push(ProbeItem {
                name,
                passed: value == expected,
                detail: format!("got {value:?}, expected {expected:?}"),
            }),
            Err(e) => items.push(ProbeItem {
                name,
                passed: false,
                detail: format!("async result could not be read back: {e}"),
            }),
        },
        Err(e) => items.push(ProbeItem {
            name,
            passed: false,
            detail: format!("failed: {e}"),
        }),
    }
}

/// The ES version the pinned engine actually supports, from the probe
/// data. Kept in code (not just the README) so a future probe run can
/// re-derive it: ES2020 core + modules + top-level await is the floor.
pub const PROBE_ES_FLOOR: &str = "ES2020";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::JsError;

    /// Print the probe table. Run with `-- --nocapture` to see it.
    #[test]
    fn print_probe_table() {
        let items = probe_engine();
        println!("probe results (rquickjs 0.14 / QuickJS-NG):");
        for item in &items {
            println!(
                "  {:<28} {:<4} {}",
                item.name,
                if item.passed { "PASS" } else { "FAIL" },
                item.detail
            );
        }
        let passed = items.iter().filter(|i| i.passed).count();
        println!("  {}/{} passed", passed, items.len());
    }

    #[test]
    fn es2020_core_passes() {
        let items = probe_engine();
        let find = |name: &str| {
            items
                .iter()
                .find(|i| i.name == name)
                .unwrap_or_else(|| panic!("probe item {name} missing"))
        };
        for name in [
            "modules",
            "top-level-await",
            "async-generators",
            "bigint",
            "proxy",
            "test262: promise-ordering",
            "test262: async-await",
            "test262: classes",
            "test262: generators",
            "test262: destructuring",
            "test262: optional-chaining",
            "test262: nullish-coalescing",
            "test262: promise-all",
        ] {
            let item = find(name);
            assert!(item.passed, "{name} should pass: {}", item.detail);
        }
    }

    #[test]
    fn probe_is_deterministic_across_runs() {
        let first = probe_engine();
        let second = probe_engine();
        assert_eq!(first.len(), second.len());
        for (a, b) in first.iter().zip(second.iter()) {
            assert_eq!(a.name, b.name);
            assert_eq!(a.passed, b.passed, "probe item {} flipped", a.name);
        }
    }

    #[test]
    fn probe_covers_the_open_questions() {
        // The report's open questions must be *answered by data*, not
        // dropped: these rows exist even when they fail.
        let items = probe_engine();
        for name in ["temporal", "array-fromAsync", "decorators"] {
            assert!(
                items.iter().any(|i| i.name == name),
                "probe must report on {name}"
            );
        }
    }

    /// A probe failure must not poison the realm for later items: an
    /// item that throws (here, via an explicit throw) is recorded and
    /// the probe continues.
    #[test]
    fn probe_survives_a_throwing_item() {
        let mut realm = TabRealm::new(&RealmConfig::default()).unwrap();
        let err: Result<JsValue, JsError> =
            realm.eval(EvalKind::Classic, "throw new Error('probe-boom')", "probe.js");
        assert!(matches!(err, Err(JsError::Exception(_))));
        assert_eq!(
            realm.eval(EvalKind::Classic, "6 * 7", "probe.js").unwrap(),
            JsValue::Number(42.0)
        );
    }
}
