"""Optional, framework-specific adapters for PurposeGuard.

The core package (everything in ``purposeguard`` *outside* this subpackage)
imports **no** third-party framework — that guarantee is what keeps the base
install dependency-light and safe to drop into any project (guardrails #2/#3).
Everything that knows about a specific framework lives here, and every framework
import is done lazily *inside* the adapter that needs it, so importing this
package never pulls in mem0, langchain, etc.

Start with :func:`guard_store` (the raw, bring-your-own-store reference). The
framework-specific adapters are specialisations of the same translate-check-tag
pattern, added one at a time.
"""

from __future__ import annotations

from .langchain import GuardedChatMessageHistory, guard_chat_history
from .mem0 import GuardedMemory, guard_mem0, guarded_memory
from .owasp_amg import CombinedVerdict, ComposedGuard, composed_guard, guard_with_amg
from .raw import GuardedStore, Store, guard_store

__all__ = [
    "guard_store",
    "GuardedStore",
    "Store",
    "guard_mem0",
    "guarded_memory",
    "GuardedMemory",
    "guard_chat_history",
    "GuardedChatMessageHistory",
    "guard_with_amg",
    "composed_guard",
    "ComposedGuard",
    "CombinedVerdict",
]
