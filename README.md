VD - Edit directory with vim
===============================================================================

Heavily inspired by `vidir` from [moreutils](https://joeyh.name/code/moreutils/) and
[edir](https://github.com/bulletmark/edir).

With a few different design choices:

*   `vd` always prompts before applying changes

*   `vd` don't integrate with `git`

*   `vd` displays an (arguably) pretty-looking preview of the changes

*   `vd` pads sequence number to same width for visual block operations

*   `vd` references `LS_COLORS` for coloring

*   `vd` supports stage/unstage items dynamically

*   `vd` treats symbolic links as files instead of resolving them

*   `vd` supports editing symbolic links


## Examples (shell)

Edit the current diretory:
```console
$ vd
```

Manage certain files:
```console
$ vd *.txt
```

Manage found files:
```console
$ fd --type f | vd
```


## Examples (vim)
A `vd`'s vim session looks like this:

```console
# ═════════════════════════════════════════════════════════════════════════════
# - Add a path to track it.
# - Sort the paths as you want.
# - Add a '#' before the id to untrack an item.
# - Add a '+' before the id to expand non-hidden items under the directory.
# - Add a '*' before the id to expand all items under the directory.
# - Add a '@' before the id to resolve the soft link.
# - Setup default vd.vimrc with
#   $ vd --vimrc
# ═════════════════════════════════════════════════════════════════════════════

110 ││ LICENSE
120 ││ README.md
131 ││ __pycache__/
140 ││ pyproject.toml
151 ││ vd/
```

Edit the content, save and quit, and `vd` prompts you the changes like this:

```
[info] Delete:[LICENSE]
[info] Rename:[README.mdd]
Continue? [(Y)es / (n)o / (e)dit / (r)edo] _
```

*   `y` to apply and continue with another `vim` session
*   `enter` to apply and quit
*   `n` to cancel the edit
*   `ctrl`+`c` to cancel the edit if you're scared of the change
*   `e` to continue editing
*   `r` to restart with initial contents
