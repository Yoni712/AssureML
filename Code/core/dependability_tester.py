from adapters.text_adapter import TextAdapter
class DependabilityTester:
    def __init__(self, model, input_adapter, output_adapter, modality):
        """
        modality: 'tabular', 'text', or 'image'
        """

        self.model = model
        self.input_adapter = input_adapter
        self.output_adapter = output_adapter
        self.modality = modality

        # Import metric functions
        if modality == "tabular":
            from metricss.robustness import robustness_test_tabular
            from metricss.consistency import consistency_test_tabular
            from metricss.variance import variance_test_tabular

            self.robustness_fn = robustness_test_tabular
            self.consistency_fn = consistency_test_tabular
            self.variance_fn = variance_test_tabular

        elif modality == "text":
            from metricss.robustness import robustness_test_text
            from metricss.consistency import consistency_test_text
            from metricss.variance import variance_test_text

            self.robustness_fn = robustness_test_text
            self.consistency_fn = consistency_test_text
            self.variance_fn = variance_test_text

        elif modality == "image":
            from metricss.robustness import robustness_test_image
            from metricss.consistency import consistency_test_image
            from metricss.variance import variance_test_image

            self.robustness_fn = robustness_test_image
            self.consistency_fn = consistency_test_image
            self.variance_fn = variance_test_image

        else:
            raise ValueError(f"Unsupported modality: {modality}")

    def evaluate(self, raw_input):
        """
        Runs all dependability tests:
        - robustness
        - consistency
        - variance
        """
        
        # 1) Apply input adapter
        model_input = self.input_adapter.adapter_input(raw_input)

        # 2) Run tests
        robustness, robustness_dist = self.robustness_fn(self.model, model_input, self.output_adapter)
        consistency, consistency_dict = self.consistency_fn(self.model, model_input, self.output_adapter)
        variance, variance_list = self.variance_fn(self.model, model_input, self.output_adapter)
        
        return {
            "modality": self.modality,
            "robustness": robustness,
            "robustness_dist": robustness_dist,
            "consistency": consistency,
            "consistency_breakdown": consistency_dict,
            "variance": variance,
            "variance_list": variance_list
        }


    def print_report(self, report):
        
        modality = report["modality"]

        print(f"\n=== DEPENDABILITY REPORT ({modality.upper()}) ===")
        print(f"Modality: {report['modality']}")
        print(f"Robustness: {report['robustness']:.4f}")
        print(f"Consistency: {report['consistency']:.4f}")
        print(f"Variance: {report['variance']:.6f}")
