from docutils import nodes


def setup(app):
    app.add_role(
        "pr", autolink("https://github.com/python/typing_extensions/pull/{}", "PR #")
    )
    app.add_role(
        "pr-cpy", autolink("https://github.com/python/cpython/pull/{}", "CPython PR #")
    )
    app.add_role(
        "issue",
        autolink("https://github.com/python/typing_extensions/issues/{}", "issue #"),
    )
    app.add_role(
        "issue-cpy",
        autolink("https://github.com/python/cpython/issues/{}", "CPython issue #"),
    )


def autolink(pattern: str, prefix: str):
    def role(name, rawtext, text: str, lineno, inliner, options=None, content=None):
        if options is None:
            options = {}
        url = pattern.format(text)
        node = nodes.reference(rawtext, f"{prefix}{text}", refuri=url, **options)
        return [node], []

    return role
