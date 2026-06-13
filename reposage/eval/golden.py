"""A hand-built golden set: natural-language questions about the Tracker repo,
each paired with the source file(s) that genuinely answer it.

This is the backbone of honest evaluation — without ground truth you can only
eyeball results. With it, you can MEASURE retrieval quality and prove that one
design (hybrid, AST chunking) beats another with numbers.

To eval a different repo, index it and rewrite this list for that codebase.
"""

GOLDEN = [
    {"q": "How is JWT authentication verified on protected routes?",
     "files": {"middleware/auth.js"}},
    {"q": "How are user passwords hashed and compared at login?",
     "files": {"models/User.js"}},
    {"q": "How are atomic transfers between accounts implemented?",
     "files": {"controllers/accounts.js"}},
    {"q": "How is a transaction's amount applied to an account balance?",
     "files": {"controllers/transactions.js"}},
    {"q": "What security middleware does the Express app use?",
     "files": {"app.js"}},
    {"q": "How do user registration and login endpoints work?",
     "files": {"controllers/auth.js"}},
    {"q": "How is spending summarised per category with an aggregation pipeline?",
     "files": {"controllers/transactions.js"}},
    {"q": "How does the React frontend store and persist the auth token?",
     "files": {"client/src/context/AuthState.js"}},
    {"q": "How are the transaction API routes defined and protected?",
     "files": {"routes/transactions.js"}},
    {"q": "How is the connection to MongoDB established?",
     "files": {"config/db.js"}},
]
