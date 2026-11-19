"""
sql — Runtime shared by the SQL backends (Postgres and SQLite).

One implementation serves the graph, the storage and the freeze axes. The differences between
engines live in :class:`~scinr.newton.sql.dialect.Dialect`; connections in
:mod:`scinr.newton.sql.engine`; the schema lifecycle in :mod:`scinr.newton.sql.schema`.

Nothing in this package imports ``sqlalchemy`` or a database driver when ``scinr.newton`` is
imported: the SQL extras (``scinr[postgres]`` / ``scinr[sqlite]``) are optional.
"""
