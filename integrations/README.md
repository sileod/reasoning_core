# Standalone integration applications

These distributions depend on reasoning-core and have their own build/install
configuration. They are shipped in the repository/source distribution, not inside
the reasoning-core wheel:

- [OpenEnv](openenv/reasoning_core_env/README.md)
- [Prime Intellect](primeintellect/reasoning_core_env/README.md)

Python adapters are in [`reasoning_core/integrations/`](../reasoning_core/integrations/README.md).
Run application-specific install and container commands from the application's
own directory. The relative layout inside each application is unchanged.
