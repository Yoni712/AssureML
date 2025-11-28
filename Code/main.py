import os
from core.dependability_tester import DependabilityTester
from models.text_model import TextModel
from models.tabular_model import TabularModel
from models.image_model import ImageModel
from adapters.text_adapter import TextAdapter
from adapters.tabular_adapter import TabularAdapter
from adapters.image_adapter import ImageAdapter
from adapters.output_adapter import OutputAdapter
from metricss.robustness import robustness_test_text, robustness_test_image, robustness_test_tabular
from metricss.consistency import consistency_test_text, consistency_test_image, consistency_test_tabular
from metricss.variance import variance_test_text, variance_test_image, variance_test_tabular
from visualization.plotter import Plotter


def create_example_digit():
    from torchvision import datasets, transforms
    import torchvision.utils as vutils

    transform = transforms.Compose([transforms.ToTensor()])

    dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    img_tensor, label = dataset[0]  # the first test digit

    vutils.save_image(img_tensor, "example_digit.png")

    print("Saved example_digit.png (digit:", label, ")")

def run_tabular_demo():
    model = TabularModel()
    tester = DependabilityTester(
        model=model,
        input_adapter=TabularAdapter(),
        output_adapter=OutputAdapter(),
        modality="tabular"
    )

    sample = [5.1, 3.5, 1.4, 0.2]
    report = tester.evaluate(sample)
    tester.print_report(report)

def run_image_demo():
    model = ImageModel()
    tester = DependabilityTester(
        model=model,
        input_adapter=ImageAdapter(),
        output_adapter=OutputAdapter(),
        modality="image"
    )

    image_path = "example_digit.png"
    report = tester.evaluate(image_path)
    tester.print_report(report)

    # ---- Visualization ----
    plotter = Plotter()

    # 1. Histogram of robustness values
    plotter.plot_robustness_distribution(report["robustness_dist"])

    # 2. Bar chart of consistency scores
    plotter.plot_consistency_breakdown(report["consistency_breakdown"])

    # 3. Line plot of variance values
    plotter.plot_variance_curve(report["variance_list"])



def run_text_demo():
    model = TextModel()
    tester = DependabilityTester(
        model=model,
        input_adapter=TextAdapter(),
        output_adapter=OutputAdapter(),
        modality="text"
    )

    text = "This movie was amazing!"
    report = tester.evaluate(text)
    tester.print_report(report)


def main():
    run_tabular_demo()
    run_text_demo()
    run_image_demo()

if __name__ == "__main__":
    main()
