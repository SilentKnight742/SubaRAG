# Contributing to SubaRAG

## Branch conventions

- `main` is the stable release branch.
- Create every work branch directly from `main` using one of these four types:
  - `feature/<descriptive-kebab-case>`
  - `fix/<descriptive-kebab-case>`
  - `chore/<descriptive-kebab-case>`
  - `release/<version-or-descriptor>`
- Merge completed, verified work back into `main` through review.

## Commit conventions

- Keep every commit focused on one coherent change.
- Use concise conventional subjects such as `feat:`, `fix:`, `docs:`, `test:`,
  and `chore:`. Add a scope when it improves clarity.
- Run the relevant checks before committing or pushing.
- Never commit credentials, API keys, local environment files, or generated
  secrets.
- Push and deployment are explicit release actions and are not implied by local
  implementation work.
