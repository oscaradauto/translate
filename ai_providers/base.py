from abc import ABC, abstractmethod


class AIProvider(ABC):
    """Contrato que debe cumplir cualquier proveedor de IA (Gemini, GitHub Models, etc.)."""

    @abstractmethod
    def warmup(self):
        """Precalienta la conexión/modelo. Debe manejar sus propios errores internamente."""
        raise NotImplementedError

    @abstractmethod
    def generate(self, system_prompt: str, prompt: str, temperature: float = 0.3,
                 max_tokens: int = 300) -> str:
        """
        Genera texto a partir de un prompt.
        Debe devolver SIEMPRE un string (vacío si falla), nunca lanzar excepción hacia afuera.
        """
        raise NotImplementedError