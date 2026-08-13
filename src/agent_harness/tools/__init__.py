"""Tool schemas and their runtime implementations.

`shared` and `searcher_only` hold the JSON schemas the model sees; `functions`
holds the runtime that executes them against a Mixedbread store. Schemas never
import the runtime.
"""
