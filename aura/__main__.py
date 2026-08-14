from __future__ import annotations

from time import sleep

from binding  import OneWayBinder
from debounce import Debouncer

class Example(OneWayBinder):

    def fire(self) -> None:
        print('Hello, World!')

def main() -> None:
    example: Example = Example.create()

    debouncer: Debouncer = Debouncer.create(
        example,
        1.0,
    )

    for _ in range(5):
        debouncer()
        sleep(0.25)

    sleep(1.25)

if __name__ == '__main__':
    main()
