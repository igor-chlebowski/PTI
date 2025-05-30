import os
from pathlib  import Path

from Detector.testhasy import predict, preprocess_image, remap_symbol
from MajorPredictor.main import MajorPredictor

class Crawler:
    def __init__(self, path):
        self.path = path

    def crawl(self):
        """
        Traverse the directory structure starting from the given path.
        This function collects all files and directories in a structured format.
        """
        structure = {'name': os.path.basename(self.path), 'type': 'directory', 'children': []}
        self.recursive_crawl(self.path, structure)
        return structure

    
    def recursive_crawl(self, current_path, structure):
        """
        Recursively crawl through the directory and build the structure.
        """
        print(f"Crawling through: {current_path} - leaf: {Crawler.is_leaf(current_path)}")

        if Crawler.is_leaf(current_path):
            leaf_path = os.path.join(current_path, os.listdir(current_path)[0])
            detection = self.handle_leaf(leaf_path)
            structure['children'].append({'name': os.listdir(current_path)[0], 'type': 'file', 'path': leaf_path, 'detection': detection})
            return

        for entry in os.listdir(current_path):
            full_path = os.path.join(current_path, entry)

            if os.path.isdir(full_path):
                # If it's a directory, create a new entry in the structure
                sub_structure = {'name': entry, 'type': 'directory', 'children': []}
                structure['children'].append(sub_structure)
                self.recursive_crawl(full_path, sub_structure)
            else:
                # If it's a file, add it to the current structure
                detection = self.handle_complex_expression(full_path, entry)
                structure['children'].append({'name': entry, 'type': 'file', 'path': full_path, 'detection': detection})

    @staticmethod
    def is_leaf(path):
        """
        Check if the given path is a singleton (no directories exit in there).
        """
        return not any(p.is_dir() for p in Path(path).iterdir())
        # list_dir = (file for file in os.listdir(path) if os.path.isdir(os.path.join(path, file)))
        # print(f"Checking if {path} is a leaf: {list(list_dir)}")
        # return any(os.path.isdir(os.path.join(path, entry)) for entry in os.listdir(path))

    def handle_leaf(self, leaf_path):
        """
        Handle the leaf node (file) in the directory structure.
        This function runs Igor's ML model for character recognition on the file.
        """
        
        try:
            print("-----------Starting prediction!")
            digit, conf = predict(leaf_path)
            char = remap_symbol(int(digit))
            print(f"Rozpoznana cyfra: {char} (pewność: {conf:.2%})")
            return char
        except Exception as e:
            print("Błąd podczas predykcji:", e)
            return '?'

    def handle_complex_expression(self, path, expression):
        """
        Handle complex expressions shown in image.
        This function runs  Leon's ML model for major operand recognition.
        """

        predict = MajorPredictor(path).predict()
        if predict is None:
            return '?'
        print(f"Rozpoznana operacja: {predict}")

        return '?'


