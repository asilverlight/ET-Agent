import asyncio
import ast
from typing import Tuple
from asyncio import BoundedSemaphore

from .base_tool import BaseTool


class PythonTool(BaseTool):
    """Execute Python Code"""

    def __init__(self, conda_env: str, max_concurrent: int = 10):
        self.python_path = conda_env
        self.semaphore = BoundedSemaphore(max_concurrent)

    @property
    def name(self) -> str:
        return "python"

    @property
    def trigger_tag(self) -> str:
        return "python"

    async def execute(self, code: str, timeout: int = 1000) -> str:
        """Execute python code and return the results"""
        async with self.semaphore:
            result, report = await self._run_code(code, timeout)
            return result if report == "Done" else report

    async def _run_code(self, code: str, timeout: int) -> Tuple[str, str]:
        code = self._preprocess_code(code)

        try:
            proc = await asyncio.create_subprocess_exec(
                self.python_path,
                "-c",
                code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
                if proc.returncode == 0:
                    return stdout.decode().strip(), "Done"
                else:
                    return "", stderr.decode().strip()
            except asyncio.TimeoutError:
                proc.kill()
                return "", "Execution Timeout"

        except Exception as e:
            return "", f"Error: {str(e)}"

    def _preprocess_code(self, code: str) -> str:
        """Make the last Python statement to be `print` statement"""
        try:
            tree = ast.parse(code)
            if tree.body:
                last_expr = tree.body[-1]
                if isinstance(last_expr, ast.Expr):
                    if not (
                        isinstance(last_expr.value, ast.Call)
                        and isinstance(last_expr.value.func, ast.Name)
                        and last_expr.value.func.id == "print"
                    ):
                        print_call = ast.Expr(
                            value=ast.Call(
                                func=ast.Name(id="print", ctx=ast.Load()),
                                args=[last_expr.value],
                                keywords=[],
                            )
                        )
                        tree.body[-1] = print_call
                        code = ast.unparse(tree)
        except:
            pass

        return code
