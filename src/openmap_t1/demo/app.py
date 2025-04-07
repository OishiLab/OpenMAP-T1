from openmap_t1.demo.controller import DemoController
from openmap_t1.demo.model import DemoModel
from openmap_t1.demo.view import DemoView


def run():
    controller = DemoController(model=DemoModel(), view=DemoView())
    controller.run()


if __name__ == "__main__":
    run()
