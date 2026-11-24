# Bundled nav-bar icons

The three toolbar icons (Back, Forward, Reload), as SVGs compiled into
the Pluto binary with `include_bytes!`.

## Why bundled

`cosmic::widget::icon::from_name` only searches the Cosmic icon theme.
On a machine without Cosmic icons installed (e.g. KDE/Breeze), the
lookup finds nothing and the button silently draws blank — no error,
no placeholder. Bundling the SVGs makes the icons render wherever
Pluto runs, and `.symbolic(true)` tints them to the theme's icon
color (white on the dark nav bar).

## Source

`go-previous-symbolic.svg`, `go-next-symbolic.svg`,
`view-refresh-symbolic.svg` — from
[pop-os/cosmic-icons](https://github.com/pop-os/cosmic-icons)
(`freedesktop/scalable/actions/`), licensed
[CC-BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
Unmodified.
