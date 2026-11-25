//! Form element APIs + constraint validation + `FormData`
//! (Stage 13 Rec 5/7/8, M5 forms).
//!
//! The element methods live on `Element.prototype` (installed by
//! [`install_form_methods`]); they no-op / return null for
//! non-controls, and throw `TypeError` for form-only methods
//! called on a non-`<form>`. All live state goes through the
//! [`DomHost`] form methods, which the production host and the
//! test host both back with the real `pluto-html` `Dom` — so the
//! dirty-value split (typed value survives `setAttribute`) holds
//! identically in tests and in the browser.
//!
//! `FormData` (installed by [`install_form_data`]) follows the
//! `Image()` pattern: a Rust factory in the registry builds the
//! entry list from the host, the JS prototype methods manipulate
//! the instance's `__entries` array in JS.

use std::cell::RefCell;
use std::rc::Rc;

use rquickjs::function::{Opt, This};
use rquickjs::{Array, Coerced, Ctx, Function, Object, Value};

use rquickjs::FromJs;

use crate::bindings::events::dispatch_to_target;
use crate::bindings::{
    BindingState, dom_key_of_this, dom_key_of_this_opt, install_entries_iterator, js, js_null,
    registry, throw_type_error, to_dom_string,
};
use crate::bindings::dom::{
    def_accessor, def_getter, def_method, elements_array, host_of, wrap_node,
};
use crate::host::{DomHost, DomKey};

// ------------------------------------------------------------------ validity

/// One control's constraint-validation state (Stage 13 Rec 8).
#[derive(Debug, Clone, Default)]
pub struct Validity {
    pub value_missing: bool,
    pub type_mismatch: bool,
    pub pattern_mismatch: bool,
    pub too_long: bool,
    pub too_short: bool,
    pub range_underflow: bool,
    pub range_overflow: bool,
    pub step_mismatch: bool,
    pub bad_input: bool,
    pub custom_error: bool,
    /// Human-readable message for the first failing constraint
    /// (the custom message when `custom_error`).
    pub message: String,
}

impl Validity {
    pub fn valid(&self) -> bool {
        !(self.value_missing
            || self.type_mismatch
            || self.pattern_mismatch
            || self.too_long
            || self.too_short
            || self.range_underflow
            || self.range_overflow
            || self.step_mismatch
            || self.bad_input
            || self.custom_error)
    }
}

/// Whether the control participates in constraint validation at
/// all (spec `willValidate`): disabled controls and buttons /
/// hidden / file / image inputs do not.
pub fn will_validate(host: &dyn DomHost, key: DomKey) -> bool {
    if host.is_disabled(key) {
        return false;
    }
    match host.tag_name(key).as_deref() {
        Some("textarea") | Some("select") => true,
        Some("input") => !matches!(
            host.input_type(key).as_deref(),
            Some("hidden" | "submit" | "button" | "reset" | "image" | "file")
        ),
        _ => false,
    }
}

/// Compute one control's validity from its attributes + live
/// state (Stage 13 Rec 8). `pattern` is evaluated with the
/// `regex` crate as a full match (`^(?:pattern)$`); an invalid
/// pattern is ignored, per spec.
pub fn validity_of(host: &dyn DomHost, key: DomKey) -> Validity {
    let mut v = Validity::default();
    if !will_validate(host, key) {
        return v;
    }
    if let Some(custom) = host.custom_validity(key) {
        if !custom.is_empty() {
            v.custom_error = true;
            v.message = custom;
            return v;
        }
    }
    let tag = host.tag_name(key).unwrap_or_default();
    let ty = host.input_type(key).unwrap_or_else(|| "text".to_string());
    let required = host.get_attribute(key, "required").is_some();
    let value = host.form_value(key).unwrap_or_default();
    let checked = host.form_checked(key).unwrap_or(false);

    // valueMissing: required + empty (unchecked box/radio, no
    // selection, empty text).
    v.value_missing = required
        && match tag.as_str() {
            "select" => host
                .form_selected_index(key)
                .map(|i| i < 0)
                .unwrap_or(true),
            "input" if ty == "checkbox" || ty == "radio" => !checked,
            _ => value.is_empty(),
        };
    if v.value_missing {
        v.message = "Please fill out this field.".to_string();
        return v;
    }
    if value.is_empty() {
        return v;
    }
    // typeMismatch: email / url.
    if tag == "input" && ty == "email" {
        let parts: Vec<&str> = value.split('@').collect();
        if parts.len() != 2 || parts[0].is_empty() || parts[1].is_empty()
            || !parts[1].contains('.')
        {
            v.type_mismatch = true;
            v.message = "Please enter an email address.".to_string();
            return v;
        }
    }
    if tag == "input" && ty == "url" {
        match url::Url::parse(&value) {
            Ok(u) if u.has_host() => {}
            _ => {
                v.type_mismatch = true;
                v.message = "Please enter a URL.".to_string();
                return v;
            }
        }
    }
    // patternMismatch (full match).
    if let Some(pattern) = host.get_attribute(key, "pattern") {
        if let Ok(re) = regex::Regex::new(&format!("^(?:{pattern})$")) {
            if !re.is_match(&value) {
                v.pattern_mismatch = true;
                v.message = "Please match the requested format.".to_string();
                return v;
            }
        }
    }
    // tooLong / tooShort (char counts).
    let len = value.chars().count();
    if let Some(max) = host
        .get_attribute(key, "maxlength")
        .and_then(|m| m.parse::<usize>().ok())
    {
        if len > max {
            v.too_long = true;
            v.message = format!("Please shorten this text to {max} characters or less.");
            return v;
        }
    }
    if let Some(min) = host
        .get_attribute(key, "minlength")
        .and_then(|m| m.parse::<usize>().ok())
    {
        if len < min {
            v.too_short = true;
            v.message = format!("Please lengthen this text to {min} characters or more.");
            return v;
        }
    }
    // rangeUnderflow / rangeOverflow / stepMismatch (numbers).
    if tag == "input" && (ty == "number" || ty == "range") {
        if let Ok(num) = value.parse::<f64>() {
            if let Some(min) = host
                .get_attribute(key, "min")
                .and_then(|m| m.parse::<f64>().ok())
            {
                if num < min {
                    v.range_underflow = true;
                    v.message = format!("Value must be {min} or more.");
                    return v;
                }
            }
            if let Some(max) = host
                .get_attribute(key, "max")
                .and_then(|m| m.parse::<f64>().ok())
            {
                if num > max {
                    v.range_overflow = true;
                    v.message = format!("Value must be {max} or less.");
                    return v;
                }
            }
            if let Some(step) = host
                .get_attribute(key, "step")
                .and_then(|s| s.parse::<f64>().ok())
                .filter(|s| *s > 0.0)
            {
                let base = host
                    .get_attribute(key, "min")
                    .and_then(|m| m.parse::<f64>().ok())
                    .unwrap_or(0.0);
                let rem = ((num - base) / step).fract().abs();
                if rem > 1e-9 && (1.0 - rem) > 1e-9 {
                    v.step_mismatch = true;
                    v.message = "Please enter a valid value.".to_string();
                    return v;
                }
            }
        } else {
            v.bad_input = true;
            v.message = "Please enter a number.".to_string();
            return v;
        }
    }
    // Dates compare lexicographically (ISO-8601).
    if tag == "input" && matches!(ty.as_str(), "date" | "month" | "time" | "datetime-local") {
        if let Some(min) = host.get_attribute(key, "min") {
            if value.as_str() < min.as_str() {
                v.range_underflow = true;
                v.message = format!("Value must be {min} or later.");
                return v;
            }
        }
        if let Some(max) = host.get_attribute(key, "max") {
            if value.as_str() > max.as_str() {
                v.range_overflow = true;
                v.message = format!("Value must be {max} or earlier.");
                return v;
            }
        }
    }
    v
}

/// The first invalid control of a form, in document order.
fn first_invalid_control(host: &dyn DomHost, form: DomKey) -> Option<DomKey> {
    host.form_controls(form)
        .into_iter()
        .find(|&c| !validity_of(host, c).valid())
}

/// Dispatch a plain event on `key`'s wrapper and return whether
/// its default action was NOT prevented.
fn dispatch_plain<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
    key: DomKey,
    ty: &str,
    bubbles: bool,
    cancelable: bool,
) -> rquickjs::Result<bool> {
    let event: Object = ctx.eval(format!(
        "new Event({ty:?}, {{ bubbles: {bubbles}, cancelable: {cancelable} }})"
    ))?;
    let id = match bindings.borrow().wrapper_ids.get(&key).copied() {
        Some(id) => id,
        None => return Ok(true),
    };
    dispatch_to_target(ctx, bindings, id, &event, ty)
}

/// The validity object for `key` (a plain object with the
/// `ValidityState` flags + `valid`).
fn validity_object<'js>(
    ctx: &Ctx<'js>,
    host: &dyn DomHost,
    key: DomKey,
) -> rquickjs::Result<Object<'js>> {
    let v = validity_of(host, key);
    let obj = Object::new(ctx.clone())?;
    obj.set("valueMissing", v.value_missing)?;
    obj.set("typeMismatch", v.type_mismatch)?;
    obj.set("patternMismatch", v.pattern_mismatch)?;
    obj.set("tooLong", v.too_long)?;
    obj.set("tooShort", v.too_short)?;
    obj.set("rangeUnderflow", v.range_underflow)?;
    obj.set("rangeOverflow", v.range_overflow)?;
    obj.set("stepMismatch", v.step_mismatch)?;
    obj.set("badInput", v.bad_input)?;
    obj.set("customError", v.custom_error)?;
    obj.set("valid", v.valid())?;
    Ok(obj)
}

// ------------------------------------------------------------------ helpers

fn tag_of<'js>(
    ctx: &Ctx<'js>,
    b: &Rc<RefCell<BindingState>>,
    this: &Object<'js>,
) -> rquickjs::Result<Option<String>> {
    let Some(key) = dom_key_of_this_opt(ctx, b, this)? else {
        return Ok(None);
    };
    let host = match host_of(b) {
        Ok(h) => h,
        Err(_) => return Ok(None),
    };
    Ok(host.borrow().tag_name(key))
}

fn is_control_tag(tag: Option<&str>) -> bool {
    matches!(tag, Some("input" | "textarea" | "select"))
}

/// The selected option's value for a `<select>` (the `value`
/// attribute, else the option's collapsed text).
fn option_value_of(host: &dyn DomHost, option: DomKey) -> String {
    if let Some(v) = host.get_attribute(option, "value") {
        return v;
    }
    host.text_content(option)
        .map(|t| t.split_whitespace().collect::<Vec<_>>().join(" "))
        .unwrap_or_default()
}

// ------------------------------------------------------------------ install

/// Install the form APIs on `Element.prototype` (Stage 13 Rec 5/7/8).
pub fn install_form_methods<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
    element: &Object<'js>,
) -> Result<(), String> {
    // .value / .defaultValue.
    let b = Rc::clone(bindings);
    let get_value = move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
        let Some(key) = dom_key_of_this_opt(&ctx, &b, &this)? else {
            return Ok(js_null(&ctx));
        };
        let host = match host_of(&b) {
            Ok(h) => h,
            Err(_) => return Ok(js_null(&ctx)),
        };
        let host = host.borrow();
        Ok(match host.form_value(key) {
            Some(v) => js(&ctx, v),
            None => js_null(&ctx),
        })
    };
    let b = Rc::clone(bindings);
    let set_value = move |ctx: Ctx<'js>, This(this): This<Object<'js>>, value: Value<'js>| -> rquickjs::Result<()> {
        let key = dom_key_of_this(&ctx, &b, &this)?;
        let text = to_dom_string(&ctx, &value)?;
        let host = host_of(&b).map_err(|e| throw_type_error(&ctx, &e))?;
        if !host.borrow().is_form_control(key) {
            return Ok(());
        }
        // Setting `.value` on a `<select>` selects the option
        // with that value (or deselects when none matches).
        let is_select = host.borrow().tag_name(key).as_deref() == Some("select");
        if is_select {
            let options = host.borrow().form_options(key);
            let mut found: Option<i64> = None;
            for (i, opt) in options.iter().enumerate() {
                if option_value_of((&*host.borrow()).as_ref(), *opt) == text {
                    found = Some(i as i64);
                    break;
                }
            }
            host.borrow_mut()
                .set_form_selected_index(key, found.unwrap_or(-1));
            return Ok(());
        }
        host.borrow_mut().set_form_value(key, &text);
        Ok(())
    };
    def_accessor(ctx, element, "value", get_value, set_value)?;

    let b = Rc::clone(bindings);
    let get_default_value = move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
        let Some(key) = dom_key_of_this_opt(&ctx, &b, &this)? else {
            return Ok(js_null(&ctx));
        };
        let host = match host_of(&b) {
            Ok(h) => h,
            Err(_) => return Ok(js_null(&ctx)),
        };
        Ok(match host.borrow().form_default_value(key) {
            Some(v) => js(&ctx, v),
            None => js_null(&ctx),
        })
    };
    let b = Rc::clone(bindings);
    let set_default_value = move |ctx: Ctx<'js>, This(this): This<Object<'js>>, value: Value<'js>| -> rquickjs::Result<()> {
        let key = dom_key_of_this(&ctx, &b, &this)?;
        let text = to_dom_string(&ctx, &value)?;
        let host = host_of(&b).map_err(|e| throw_type_error(&ctx, &e))?;
        if host.borrow().is_form_control(key) {
            host.borrow_mut().set_attribute(key, "value", &text);
        }
        Ok(())
    };
    def_accessor(ctx, element, "defaultValue", get_default_value, set_default_value)?;

    // .checked / .defaultChecked.
    let b = Rc::clone(bindings);
    let get_checked = move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
        let Some(key) = dom_key_of_this_opt(&ctx, &b, &this)? else {
            return Ok(js_null(&ctx));
        };
        let host = match host_of(&b) {
            Ok(h) => h,
            Err(_) => return Ok(js_null(&ctx)),
        };
        Ok(match host.borrow().form_checked(key) {
            Some(c) => js(&ctx, c),
            None => js_null(&ctx),
        })
    };
    let b = Rc::clone(bindings);
    let set_checked = move |ctx: Ctx<'js>, This(this): This<Object<'js>>, value: Value<'js>| -> rquickjs::Result<()> {
        let key = dom_key_of_this(&ctx, &b, &this)?;
        let checked = bool::from_js(&ctx, value).unwrap_or(false);
        let host = host_of(&b).map_err(|e| throw_type_error(&ctx, &e))?;
        host.borrow_mut().set_form_checked(key, checked);
        Ok(())
    };
    def_accessor(ctx, element, "checked", get_checked, set_checked)?;

    let b = Rc::clone(bindings);
    let get_default_checked = move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
        let Some(key) = dom_key_of_this_opt(&ctx, &b, &this)? else {
            return Ok(js_null(&ctx));
        };
        let host = match host_of(&b) {
            Ok(h) => h,
            Err(_) => return Ok(js_null(&ctx)),
        };
        let has = host.borrow().get_attribute(key, "checked").is_some();
        Ok(js(&ctx, has))
    };
    let b = Rc::clone(bindings);
    let set_default_checked = move |ctx: Ctx<'js>, This(this): This<Object<'js>>, value: Value<'js>| -> rquickjs::Result<()> {
        let key = dom_key_of_this(&ctx, &b, &this)?;
        let checked = bool::from_js(&ctx, value).unwrap_or(false);
        let host = host_of(&b).map_err(|e| throw_type_error(&ctx, &e))?;
        if checked {
            host.borrow_mut().set_attribute(key, "checked", "");
        } else {
            host.borrow_mut().remove_attribute(key, "checked");
        }
        Ok(())
    };
    def_accessor(ctx, element, "defaultChecked", get_default_checked, set_default_checked)?;

    // .selectedIndex / .selected / .options / .selectedOptions.
    let b = Rc::clone(bindings);
    let get_selected_index = move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
        let Some(key) = dom_key_of_this_opt(&ctx, &b, &this)? else {
            return Ok(js_null(&ctx));
        };
        let host = match host_of(&b) {
            Ok(h) => h,
            Err(_) => return Ok(js_null(&ctx)),
        };
        Ok(match host.borrow().form_selected_index(key) {
            Some(i) => js(&ctx, i as f64),
            None => js_null(&ctx),
        })
    };
    let b = Rc::clone(bindings);
    let set_selected_index = move |ctx: Ctx<'js>, This(this): This<Object<'js>>, value: Value<'js>| -> rquickjs::Result<()> {
        let key = dom_key_of_this(&ctx, &b, &this)?;
        let idx = f64::from_js(&ctx, value).unwrap_or(-1.0) as i64;
        let host = host_of(&b).map_err(|e| throw_type_error(&ctx, &e))?;
        host.borrow_mut().set_form_selected_index(key, idx);
        Ok(())
    };
    def_accessor(ctx, element, "selectedIndex", get_selected_index, set_selected_index)?;

    let b = Rc::clone(bindings);
    def_accessor(ctx, element, "selected", {
        let b2 = Rc::clone(&b);
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
            let Some(key) = dom_key_of_this_opt(&ctx, &b2, &this)? else {
                return Ok(js_null(&ctx));
            };
            let host = match host_of(&b2) {
                Ok(h) => h,
                Err(_) => return Ok(js_null(&ctx)),
            };
            let host = host.borrow();
            if host.tag_name(key).as_deref() != Some("option") {
                return Ok(js_null(&ctx));
            }
            let mut cur = host.parent(key);
            while let Some(n) = cur {
                if host.tag_name(n).as_deref() == Some("select") {
                    let options = host.form_options(n);
                    let idx = host.form_selected_index(n).unwrap_or(-1);
                    let pos = options.iter().position(|&o| o == key).map(|i| i as i64);
                    return Ok(js(&ctx, pos == Some(idx)));
                }
                cur = host.parent(n);
            }
            Ok(js(&ctx, false))
        }
    }, {
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>, value: Value<'js>| -> rquickjs::Result<()> {
            let key = dom_key_of_this(&ctx, &b, &this)?;
            let selected = bool::from_js(&ctx, value).unwrap_or(false);
            let host = host_of(&b).map_err(|e| throw_type_error(&ctx, &e))?;
            if host.borrow().tag_name(key).as_deref() != Some("option") {
                return Ok(());
            }
            let mut cur = host.borrow().parent(key);
            while let Some(n) = cur {
                if host.borrow().tag_name(n).as_deref() == Some("select") {
                    let options = host.borrow().form_options(n);
                    let pos = options.iter().position(|&o| o == key).map(|i| i as i64);
                    if selected {
                        if let Some(p) = pos {
                            host.borrow_mut().set_form_selected_index(n, p);
                        }
                    } else if host.borrow().form_selected_index(n) == pos {
                        host.borrow_mut().set_form_selected_index(n, -1);
                    }
                    break;
                }
                cur = host.borrow().parent(n);
            }
            Ok(())
        }
    })?;

    let b = Rc::clone(bindings);
    def_getter(ctx, element, "options", move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
        let Some(key) = dom_key_of_this_opt(&ctx, &b, &this)? else {
            return Ok(Value::from(Array::new(ctx.clone())?));
        };
        let host = match host_of(&b) {
            Ok(h) => h,
            Err(_) => return Ok(Value::from(Array::new(ctx.clone())?)),
        };
        elements_array(&ctx, &b, host.borrow().form_options(key))
    })?;

    let b = Rc::clone(bindings);
    def_getter(ctx, element, "selectedOptions", move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
        let Some(key) = dom_key_of_this_opt(&ctx, &b, &this)? else {
            return Ok(Value::from(Array::new(ctx.clone())?));
        };
        let host = match host_of(&b) {
            Ok(h) => h,
            Err(_) => return Ok(Value::from(Array::new(ctx.clone())?)),
        };
        let host = host.borrow();
        let idx = host.form_selected_index(key).unwrap_or(-1);
        if idx < 0 {
            return Ok(Value::from(Array::new(ctx.clone())?));
        }
        let options = host.form_options(key);
        match options.get(idx as usize) {
            Some(&opt) => elements_array(&ctx, &b, vec![opt]),
            None => Ok(Value::from(Array::new(ctx.clone())?)),
        }
    })?;

    // .form / .elements.
    let b = Rc::clone(bindings);
    def_getter(ctx, element, "form", move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
        let Some(key) = dom_key_of_this_opt(&ctx, &b, &this)? else {
            return Ok(js_null(&ctx));
        };
        let host = match host_of(&b) {
            Ok(h) => h,
            Err(_) => return Ok(js_null(&ctx)),
        };
        match host.borrow().form_owner(key) {
            Some(form) => Ok(Value::from(wrap_node(&ctx, &b, form)?)),
            None => Ok(js_null(&ctx)),
        }
    })?;

    let b = Rc::clone(bindings);
    def_getter(ctx, element, "elements", move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
        let Some(key) = dom_key_of_this_opt(&ctx, &b, &this)? else {
            return Ok(Value::from(Array::new(ctx.clone())?));
        };
        let host = match host_of(&b) {
            Ok(h) => h,
            Err(_) => return Ok(Value::from(Array::new(ctx.clone())?)),
        };
        if host.borrow().tag_name(key).as_deref() != Some("form") {
            return Ok(Value::from(Array::new(ctx.clone())?));
        }
        elements_array(&ctx, &b, host.borrow().form_controls(key))
    })?;

    // .name / .type / .disabled / .placeholder / .maxLength.
    let b = Rc::clone(bindings);
    def_getter(ctx, element, "name", move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
        let Some(key) = dom_key_of_this_opt(&ctx, &b, &this)? else {
            return Ok(js(&ctx, ""));
        };
        let host = match host_of(&b) {
            Ok(h) => h,
            Err(_) => return Ok(js(&ctx, "")),
        };
        Ok(js(&ctx, host.borrow().get_attribute(key, "name").unwrap_or_default()))
    })?;

    let b = Rc::clone(bindings);
    def_getter(ctx, element, "type", move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
        let Some(key) = dom_key_of_this_opt(&ctx, &b, &this)? else {
            return Ok(js_null(&ctx));
        };
        let host = match host_of(&b) {
            Ok(h) => h,
            Err(_) => return Ok(js_null(&ctx)),
        };
        let host = host.borrow();
        let tag = host.tag_name(key);
        let ty = match tag.as_deref() {
            Some("input") => host.input_type(key).unwrap_or_else(|| "text".to_string()),
            Some("button") => match host.get_attribute(key, "type").map(|t| t.to_ascii_lowercase()).as_deref() {
                Some("submit") | Some("reset") | Some("button") => host.get_attribute(key, "type").unwrap().to_ascii_lowercase(),
                _ => "submit".to_string(),
            },
            Some("select") => "select-one".to_string(),
            Some("textarea") => "textarea".to_string(),
            _ => return Ok(js_null(&ctx)),
        };
        Ok(js(&ctx, ty))
    })?;

    let b = Rc::clone(bindings);
    let set_disabled = move |ctx: Ctx<'js>, This(this): This<Object<'js>>, value: Value<'js>| -> rquickjs::Result<()> {
        let key = dom_key_of_this(&ctx, &b, &this)?;
        let disabled = bool::from_js(&ctx, value).unwrap_or(false);
        let host = host_of(&b).map_err(|e| throw_type_error(&ctx, &e))?;
        if disabled {
            host.borrow_mut().set_attribute(key, "disabled", "");
        } else {
            host.borrow_mut().remove_attribute(key, "disabled");
        }
        Ok(())
    };
    {
        // Re-install `disabled` as a full accessor (getter above +
        // this setter): `def_accessor` replaces the property.
        let b = Rc::clone(bindings);
        let get_disabled = move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
            let Some(key) = dom_key_of_this_opt(&ctx, &b, &this)? else {
                return Ok(js(&ctx, false));
            };
            let host = match host_of(&b) {
                Ok(h) => h,
                Err(_) => return Ok(js(&ctx, false)),
            };
            Ok(js(&ctx, host.borrow().is_disabled(key)))
        };
        def_accessor(ctx, element, "disabled", get_disabled, set_disabled)?;
    }

    let b = Rc::clone(bindings);
    def_getter(ctx, element, "placeholder", move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
        let Some(key) = dom_key_of_this_opt(&ctx, &b, &this)? else {
            return Ok(js(&ctx, ""));
        };
        let host = match host_of(&b) {
            Ok(h) => h,
            Err(_) => return Ok(js(&ctx, "")),
        };
        Ok(js(&ctx, host.borrow().get_attribute(key, "placeholder").unwrap_or_default()))
    })?;

    let b = Rc::clone(bindings);
    def_getter(ctx, element, "maxLength", move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
        let Some(key) = dom_key_of_this_opt(&ctx, &b, &this)? else {
            return Ok(js(&ctx, -1.0));
        };
        let host = match host_of(&b) {
            Ok(h) => h,
            Err(_) => return Ok(js(&ctx, -1.0)),
        };
        let host = host.borrow();
        if !is_control_tag(host.tag_name(key).as_deref()) {
            return Ok(js(&ctx, -1.0));
        }
        Ok(js(&ctx, host.get_attribute(key, "maxlength").and_then(|m| m.parse::<f64>().ok()).unwrap_or(524288.0)))
    })?;

    // selectionStart / selectionEnd / setSelectionRange / select().
    for (prop, is_start) in [("selectionStart", true), ("selectionEnd", false)] {
        let b = Rc::clone(bindings);
        let setter = move |ctx: Ctx<'js>, This(this): This<Object<'js>>, value: Value<'js>| -> rquickjs::Result<()> {
            let key = dom_key_of_this(&ctx, &b, &this)?;
            let v = f64::from_js(&ctx, value).unwrap_or(0.0).max(0.0) as usize;
            let host = host_of(&b).map_err(|e| throw_type_error(&ctx, &e))?;
            let (s, e) = host.borrow().form_selection(key).unwrap_or((0, 0));
            if is_start {
                host.borrow_mut().set_form_selection(key, v, e.max(v));
            } else {
                host.borrow_mut().set_form_selection(key, s.min(v), v);
            }
            Ok(())
        };
        // Re-install as a full accessor (getter + setter).
        let b = Rc::clone(bindings);
        let getter = move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
            let Some(key) = dom_key_of_this_opt(&ctx, &b, &this)? else {
                return Ok(js_null(&ctx));
            };
            let host = match host_of(&b) {
                Ok(h) => h,
                Err(_) => return Ok(js_null(&ctx)),
            };
            Ok(match host.borrow().form_selection(key) {
                Some((s, e)) => js(&ctx, if is_start { s as f64 } else { e as f64 }),
                None => js_null(&ctx),
            })
        };
        def_accessor(ctx, element, prop, getter, setter)?;
    }

    let b = Rc::clone(bindings);
    let set_range = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>, start: f64, end: Opt<f64>| -> rquickjs::Result<()> {
            let key = dom_key_of_this(&ctx, &b, &this)?;
            let host = host_of(&b).map_err(|e| throw_type_error(&ctx, &e))?;
            let end = end.0.unwrap_or(start);
            host.borrow_mut().set_form_selection(key, start.max(0.0) as usize, end.max(0.0) as usize);
            Ok(())
        },
    )
    .map_err(|e| format!("setSelectionRange: {e}"))?;
    def_method(ctx, element, "setSelectionRange", set_range)?;

    let b = Rc::clone(bindings);
    let select_all = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<()> {
            let key = dom_key_of_this(&ctx, &b, &this)?;
            let host = host_of(&b).map_err(|e| throw_type_error(&ctx, &e))?;
            let len = host.borrow().form_value(key).map(|v| v.chars().count()).unwrap_or(0);
            host.borrow_mut().set_form_selection(key, 0, len);
            Ok(())
        },
    )
    .map_err(|e| format!("select: {e}"))?;
    def_method(ctx, element, "select", select_all)?;

    // focus() / blur().
    let b = Rc::clone(bindings);
    let focus = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<()> {
            let key = dom_key_of_this(&ctx, &b, &this)?;
            let host = host_of(&b).map_err(|e| throw_type_error(&ctx, &e))?;
            let old = host.borrow().active_element();
            if host.borrow_mut().focus_element(key) {
                if let Some(old) = old.filter(|&o| o != key) {
                    let _ = dispatch_plain(&ctx, &b, old, "blur", false, false);
                }
                let _ = dispatch_plain(&ctx, &b, key, "focus", false, false);
            }
            Ok(())
        },
    )
    .map_err(|e| format!("focus: {e}"))?;
    def_method(ctx, element, "focus", focus)?;

    let b = Rc::clone(bindings);
    let blur = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<()> {
            let key = dom_key_of_this(&ctx, &b, &this)?;
            let host = host_of(&b).map_err(|e| throw_type_error(&ctx, &e))?;
            if host.borrow().active_element() == Some(key) {
                host.borrow_mut().blur_element();
                let _ = dispatch_plain(&ctx, &b, key, "blur", false, false);
            }
            Ok(())
        },
    )
    .map_err(|e| format!("blur: {e}"))?;
    def_method(ctx, element, "blur", blur)?;

    // Constraint validation (Rec 8).
    let b = Rc::clone(bindings);
    def_getter(ctx, element, "validity", move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
        let Some(key) = dom_key_of_this_opt(&ctx, &b, &this)? else {
            return Ok(js_null(&ctx));
        };
        let host = match host_of(&b) {
            Ok(h) => h,
            Err(_) => return Ok(js_null(&ctx)),
        };
        if !will_validate((&*host.borrow()).as_ref(), key) && host.borrow().tag_name(key).as_deref() != Some("form") {
            // Non-validatable elements have no `validity` object;
            // controls always do.
            if !is_control_tag(host.borrow().tag_name(key).as_deref()) {
                return Ok(js_null(&ctx));
            }
        }
        Ok(Value::from(validity_object(&ctx, (&*host.borrow()).as_ref(), key)?))
    })?;

    let b = Rc::clone(bindings);
    def_getter(ctx, element, "validationMessage", move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
        let Some(key) = dom_key_of_this_opt(&ctx, &b, &this)? else {
            return Ok(js(&ctx, ""));
        };
        let host = match host_of(&b) {
            Ok(h) => h,
            Err(_) => return Ok(js(&ctx, "")),
        };
        Ok(js(&ctx, validity_of((&*host.borrow()).as_ref(), key).message))
    })?;

    let b = Rc::clone(bindings);
    def_getter(ctx, element, "willValidate", move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<Value<'js>> {
        let Some(key) = dom_key_of_this_opt(&ctx, &b, &this)? else {
            return Ok(js(&ctx, false));
        };
        let host = match host_of(&b) {
            Ok(h) => h,
            Err(_) => return Ok(js(&ctx, false)),
        };
        Ok(js(&ctx, will_validate((&*host.borrow()).as_ref(), key)))
    })?;

    let b = Rc::clone(bindings);
    let set_custom = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>, message: Coerced<String>| -> rquickjs::Result<()> {
            let key = dom_key_of_this(&ctx, &b, &this)?;
            let host = host_of(&b).map_err(|e| throw_type_error(&ctx, &e))?;
            host.borrow_mut().set_custom_validity(key, &message);
            Ok(())
        },
    )
    .map_err(|e| format!("setCustomValidity: {e}"))?;
    def_method(ctx, element, "setCustomValidity", set_custom)?;

    let b = Rc::clone(bindings);
    let check_validity = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<bool> {
            let key = dom_key_of_this(&ctx, &b, &this)?;
            let host = host_of(&b).map_err(|e| throw_type_error(&ctx, &e))?;
            let host = host.borrow();
            if host.tag_name(key).as_deref() == Some("form") {
                return Ok(first_invalid_control((&*host).as_ref(), key).is_none());
            }
            Ok(validity_of((&*host).as_ref(), key).valid())
        },
    )
    .map_err(|e| format!("checkValidity: {e}"))?;
    def_method(ctx, element, "checkValidity", check_validity)?;

    let b = Rc::clone(bindings);
    let report_validity = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<bool> {
            let key = dom_key_of_this(&ctx, &b, &this)?;
            let host = host_of(&b).map_err(|e| throw_type_error(&ctx, &e))?;
            let host = host.borrow();
            if host.tag_name(key).as_deref() == Some("form") {
                if let Some(bad) = first_invalid_control((&*host).as_ref(), key) {
                    let _ = dispatch_plain(&ctx, &b, bad, "invalid", false, true);
                    return Ok(false);
                }
                return Ok(true);
            }
            let valid = validity_of((&*host).as_ref(), key).valid();
            if !valid {
                let _ = dispatch_plain(&ctx, &b, key, "invalid", false, true);
            }
            Ok(valid)
        },
    )
    .map_err(|e| format!("reportValidity: {e}"))?;
    def_method(ctx, element, "reportValidity", report_validity)?;

    // Form methods (Rec 7): submit / reset / requestSubmit.
    let b = Rc::clone(bindings);
    let submit = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<()> {
            let key = dom_key_of_this(&ctx, &b, &this)?;
            let host = host_of(&b).map_err(|e| throw_type_error(&ctx, &e))?;
            if host.borrow().tag_name(key).as_deref() != Some("form") {
                return Err(throw_type_error(&ctx, "submit() called on a non-form element"));
            }
            // No validation, no event — straight to the renderer.
            host.borrow_mut().request_form_submit(key, None);
            Ok(())
        },
    )
    .map_err(|e| format!("submit: {e}"))?;
    def_method(ctx, element, "submit", submit)?;

    let b = Rc::clone(bindings);
    let reset = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>| -> rquickjs::Result<()> {
            let key = dom_key_of_this(&ctx, &b, &this)?;
            let host = host_of(&b).map_err(|e| throw_type_error(&ctx, &e))?;
            if host.borrow().tag_name(key).as_deref() != Some("form") {
                return Err(throw_type_error(&ctx, "reset() called on a non-form element"));
            }
            // Cancelable `reset` event; prevented => no reset.
            if dispatch_plain(&ctx, &b, key, "reset", true, true)? {
                host.borrow_mut().reset_form(key);
            }
            Ok(())
        },
    )
    .map_err(|e| format!("reset: {e}"))?;
    def_method(ctx, element, "reset", reset)?;

    let b = Rc::clone(bindings);
    let request_submit = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, This(this): This<Object<'js>>, submitter: Opt<Value<'js>>| -> rquickjs::Result<()> {
            let key = dom_key_of_this(&ctx, &b, &this)?;
            let host = host_of(&b).map_err(|e| throw_type_error(&ctx, &e))?;
            if host.borrow().tag_name(key).as_deref() != Some("form") {
                return Err(throw_type_error(&ctx, "requestSubmit() called on a non-form element"));
            }
            let submitter_key = match submitter.0 {
                Some(v) if !v.is_undefined() && !v.is_null() => {
                    let obj = v.as_object().ok_or_else(|| throw_type_error(&ctx, "requestSubmit: submitter must be an element"))?;
                    dom_key_of_this_opt(&ctx, &b, obj)?.filter(|&k| {
                        let h = host.borrow();
                        h.form_owner(k) == Some(key)
                    })
                }
                _ => None,
            };
            // Validation first: an invalid form reports validity
            // on the first invalid control and never fires `submit`.
            if let Some(bad) = first_invalid_control((&*host.borrow()).as_ref(), key) {
                let _ = dispatch_plain(&ctx, &b, bad, "invalid", false, true);
                return Ok(());
            }
            // Cancelable `submit` event; prevented => no navigation.
            if dispatch_plain(&ctx, &b, key, "submit", true, true)? {
                host.borrow_mut().request_form_submit(key, submitter_key);
            }
            Ok(())
        },
    )
    .map_err(|e| format!("requestSubmit: {e}"))?;
    def_method(ctx, element, "requestSubmit", request_submit)?;

    Ok(())
}

// ------------------------------------------------------------------ FormData

/// Install the `FormData` global (Stage 13 Rec 7): a Rust factory
/// in the registry builds the entry list from the host, the JS
/// prototype methods manipulate the instance's `__entries` array.
pub fn install_form_data<'js>(
    ctx: &Ctx<'js>,
    bindings: &Rc<RefCell<BindingState>>,
) -> Result<(), String> {
    let b = Rc::clone(bindings);
    let factory = Function::new(
        ctx.clone(),
        move |ctx: Ctx<'js>, form: Value<'js>| -> rquickjs::Result<Value<'js>> {
            let entries: Vec<(String, String)> = if form.is_undefined() || form.is_null() {
                Vec::new()
            } else {
                let obj = form
                    .as_object()
                    .ok_or_else(|| throw_type_error(&ctx, "FormData: argument must be a form element"))?;
                let Some(key) = dom_key_of_this_opt(&ctx, &b, obj)? else {
                    return Err(throw_type_error(&ctx, "FormData: argument must be a form element"));
                };
                let host = host_of(&b).map_err(|e| throw_type_error(&ctx, &e))?;
                if host.borrow().tag_name(key).as_deref() != Some("form") {
                    return Err(throw_type_error(&ctx, "FormData: argument must be a form element"));
                }
                host.borrow().form_entries(key, None)
            };
            let arr = Array::new(ctx.clone())?;
            for (i, (name, value)) in entries.into_iter().enumerate() {
                let pair = Array::new(ctx.clone())?;
                pair.set(0, name)?;
                pair.set(1, value)?;
                arr.set(i, Value::from(pair))?;
            }
            Ok(Value::from(arr))
        },
    )
    .map_err(|e| format!("FormData factory: {e}"))?;
    let reg = registry(ctx).map_err(|e| format!("registry: {e}"))?;
    reg.set("formDataFactory", factory)
        .map_err(|e| format!("install FormData factory: {e}"))?;
    ctx.eval::<(), _>(
        "globalThis.FormData = function FormData(form) { \
           if (!(this instanceof FormData)) throw new TypeError('FormData constructor requires new'); \
           this.__entries = globalThis[Symbol.for('pluto.registry')].formDataFactory(form === undefined ? null : form); \
         };",
    )
    .map_err(|e| format!("install FormData: {e}"))?;
    let globals = ctx.globals();
    let ctor: Object = globals.get("FormData").map_err(|e| e.to_string())?;
    let proto: Object = ctor.get("prototype").map_err(|e| e.to_string())?;
    // append / delete / get / getAll / has / set.
    ctx.eval::<(), _>(
        "FormData.prototype.append = function (name, value) { \
           this.__entries.push([String(name), String(value)]); \
         }; \
         FormData.prototype.delete = function (name) { \
           name = String(name); \
           this.__entries = this.__entries.filter(function (p) { return p[0] !== name; }); \
         }; \
         FormData.prototype.get = function (name) { \
           name = String(name); \
           for (var i = 0; i < this.__entries.length; i++) \
             if (this.__entries[i][0] === name) return this.__entries[i][1]; \
           return null; \
         }; \
         FormData.prototype.getAll = function (name) { \
           name = String(name); \
           var out = []; \
           for (var i = 0; i < this.__entries.length; i++) \
             if (this.__entries[i][0] === name) out.push(this.__entries[i][1]); \
           return out; \
         }; \
         FormData.prototype.has = function (name) { \
           return this.get(name) !== null; \
         }; \
         FormData.prototype.set = function (name, value) { \
           name = String(name); value = String(value); \
           var seen = false; \
           this.__entries = this.__entries.filter(function (p) { \
             if (p[0] !== name) return true; \
             if (!seen) { p[1] = value; seen = true; return true; } \
             return false; \
           }); \
           if (!seen) this.__entries.push([name, value]); \
         }; \
         FormData.prototype.entries = function () { return this.__entries.slice(); }; \
         FormData.prototype.keys = function () { return this.__entries.map(function (p) { return p[0]; }); }; \
         FormData.prototype.values = function () { return this.__entries.map(function (p) { return p[1]; }); };",
    )
    .map_err(|e| format!("install FormData methods: {e}"))?;
    install_entries_iterator(ctx, &proto).map_err(|e| format!("FormData iterator: {e}"))?;
    Ok(())
}
