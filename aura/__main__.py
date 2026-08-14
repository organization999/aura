from __future__ import annotations

from time import sleep

from binding  import OneWayBinder
from debounce import Debouncer

class Example(OneWayBinder):

    def fire(self) -> None:
        print('Hello, World!')

def main() -> None:
    example: Example = Example.create()

    # Attach the backup debouncer once. The application continues to invoke
    # `example`, not the debouncer.
    Debouncer.create(example, 1.0)

    # Each access postpones Example.fire().
    for _ in range(5):
        example()
        sleep(0.25)

    # Example.fire() runs once after one full quiet second.
    sleep(1.25)


if __name__ == '__main__':
    main()
