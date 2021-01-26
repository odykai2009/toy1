"""

Description:

Customize Source class in graphviz.
"""

__version__ = '1.0'

import os
import errno
import subprocess
from graphviz.files import Source
from graphviz._compat import text_type
from mltemplate.mlutils import is_not_none
from mltemplate.mlutils import merge_dicts


class MLSource(Source):
    """
        Override Graphviz Source class pipe method to accept env variable
        - This subclass is used to bypass our OpalServer AppServer
            not having Graphviz engines in system path env var
        - Rather than append the executable path to the sys path env var we decided to
            embed the sys path by in our mlmodel.
        - Error: Failed to build model: failed to execute ['dot', '-Tpng'] ...
    """
    def pipe(self, format=None, env=None):
        """Return the source piped through the Graphviz layout command.

        Args:
            :param format: The output format used for rendering ('pdf', 'png', etc.).
            :param env: Custom environment variables passed into subprocess command
        Returns:
            :return: Stdout of the layout command.
            :raise RuntimeError: On OsError, if ENOENT:
                    RuntimeError because Graphiv executables not on sys path
        """
        if format is None:
            format = self._format

        env = merge_dicts(*filter(is_not_none, (os.environ, env)))
        cmd = self._cmd(self._engine, format)
        data = text_type(self.source).encode(self._encoding)

        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
        except OSError as e:
            if e.errno == errno.ENOENT:
                raise RuntimeError('failed to execute %r, '
                                   'make sure the Graphviz executables '
                                   'are on your systems\' path' % cmd)
            else:  # pragma: no cover
                raise

        outs, _ = proc.communicate(data)

        return outs

    def render(self, filename=None, directory=None, view=False, cleanup=False, env=None):
        """Save the source to file and render with the Graphviz engine.
        Args:
            :param filename: Filename for saving the source (defaults to name + '.gv')
            :param directory: (Sub)directory for source saving and rendering.
            :param view: Open the rendered result with the default application.
            :param cleanup: Delete the source file after rendering.
            :param env: Custom environment variables passed into subprocess command
        Returns:
            :return: The (possibly relative) path of the rendered file.
            :raise: RuntimeError - If can't find graphviz executable on system
        """
        filepath = self.save(filename, directory)
        env = merge_dicts(*filter(is_not_none, (os.environ, env)))
        cmd = self._cmd(self._engine, self._format, filepath)

        try:
            proc = subprocess.Popen(cmd, env=env)
        except OSError as e:
            if e.errno == errno.ENOENT:
                raise RuntimeError(
                    "".join(
                        [
                            'failed to execute %r, ',
                            'make sure the Graphviz executables ',
                            'are on your systems\' path',
                        ]
                    ) % cmd
                )
            else:  # pragma: no cover
                raise

        returncode = proc.wait()

        if cleanup:
            os.remove(filepath)

        rendered = '%s.%s' % (filepath, self._format)

        if view:
            self._view(rendered, self._format)

        return rendered
