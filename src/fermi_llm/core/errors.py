"""Error types shared by the core and every component."""


class FermiLLMError(Exception):
    """Base class for errors raised by the platform."""


class ComponentError(FermiLLMError):
    """A component could not be registered, resolved or constructed."""


class DuplicateComponent(ComponentError):
    """Two components claim the same (kind, name) without one replacing the other."""


class UnknownComponent(ComponentError):
    """A component was requested by a name nothing registered."""


class PluginLoadError(FermiLLMError):
    """A plugin module raised while being imported or registered."""


class ContractError(FermiLLMError):
    """A component does not satisfy the contract of its kind."""
