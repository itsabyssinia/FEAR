from chgnet.model import CHGNet
from ase.calculators.calculator import Calculator, all_changes
from pymatgen.io.ase import AseAtomsAdaptor



class CHGNetCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, model=None, **kwargs):
        super().__init__(**kwargs)
        self.model = model or CHGNet.load()

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        structure = AseAtomsAdaptor.get_structure(atoms)
        results = self.model.predict_structure(structure)

        #print("CHGNet Prediction Results:", results)  

        self.results = {
            "energy": results["e"],     
            "forces": results["f"],     
        }
