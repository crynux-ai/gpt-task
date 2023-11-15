from annotated_types import MinLen
from typing_extensions import Annotated

NonEmptyString = Annotated[str, MinLen(1)]
