import os
from overrides import override

import plotly.io as pio
import plotly.graph_objects as go

pio.kaleido.scope.default_format = "png"
pio.kaleido.scope.default_width = 1000
pio.kaleido.scope.default_height = 600

LATEX_BASE_DIR = "/home/nkuechen/Documents/Thesis/latex/Bachelor Thesis"


class LatexFigure:
    """Base class for a LatexFigure"""

    _begin_type = "figure"
    _tab_prefix = ""
    _label_prefix = "fig"

    def __init__(
        self,
        resource_path: str,
        figure: go.Figure,
        caption: str | None = None,
        label: str | None = None,
    ) -> None:
        """Initializer function for the class.

        :param resource_path: The path where the PNG-image should be saved under in the LATEX_BASE_DIR directory
        :param figure: The plotly.graph_object.Figure object that is the figure
        :param caption: The caption of the figure in latex, if this is None the figure will not have a caption, defaults to None
        :param label: The label of the figure in latex, if this is None the figure will not have a label, defaults to None
        """
        self.resource_path = resource_path
        self.figure = figure
        self.caption = caption
        self.label = label

    def generate_latex_code(self, width: float = 1.0) -> str:
        """Generates the latex code for the figure

        :param width: The width of the figure in latex, defaults to 1.0
        :return: The latex code
        """
        latex_code = f"{self._tab_prefix}\\begin{{{self._begin_type}}}{{{width:.2f}\\linewidth}}\n"
        latex_code += f"{self._tab_prefix}\\includegraphics[width=\\linewidth]{{{self.resource_path}}}\n"
        if self.caption:
            latex_code += f"{self._tab_prefix}\\caption{{{self.caption}}}\n"
        if self.label:
            latex_code += (
                f"{self._tab_prefix}\\label{{{self._label_prefix}:{self.label}}}\n"
            )
        latex_code += f"{self._tab_prefix}\\end{{{self._begin_type}}}\n"
        return latex_code

    def write_latex_code_to_file(
        self, file_name: str, base_dir: str = LATEX_BASE_DIR, width: float = 1.0
    ) -> None:
        """Generates and writes the latex code to a file

        :param file_name: The name of the file to write to
        :param base_dir: The base directory for the file, defaults to LATEX_BASE_DIR
        :param width: The width of the latex figure, defaults to 1.0
        """
        file_path = os.path.join(base_dir, file_name)
        latex_code = self.generate_latex_code(width=width)
        print(f"Writing latex code to {file_path}")
        with open(file_path, "w") as file:
            file.write(latex_code)
        print("Done!")

    def save_figure(self, local_directory: str = LATEX_BASE_DIR) -> None:
        """Saves the figure as a PNG file to the local directory under its resource path

        :param local_directory: The local directory to save the figure in, defaults to LATEX_BASE_DIR
        """
        subfig_path = os.path.join(local_directory, self.resource_path)
        directory = os.path.dirname(subfig_path)
        if not os.path.exists(directory):
            print(f'The directory "{directory}" does not exist. We will create it.')
            os.makedirs(directory)
        try:
            print(f'Saving subfigure to "{subfig_path}"...')
            pio.write_image(self.figure, subfig_path)
            print("Done!")
        except ValueError:
            print(f'Error saving subfigure to "{subfig_path}". This figure was None.')


class LatexSubfigure(LatexFigure):
    """A subclass for latex subfigures"""

    _begin_type = "subfigure"
    _tab_prefix = "    "
    _label_prefix = "subfig"


class LatexSubfigureGrid(LatexFigure):
    """A subclass for grids of subfigures"""

    @override
    def __init__(
        self,
        caption: str | None = None,
        label: str | None = None,
    ) -> None:
        """Initializes the subfigure grid

        :param caption: The caption of the entire figure, if None there will not be a caption, defaults to None
        :param label: The label of the subfigure grid, if None there will not be a label, defaults to None
        """
        self.subfigures = [[]]
        self.caption = caption
        self.label = label

    def add_newline(self) -> None:
        """Adds a newline between two subfigures. This will end one row of subfigures."""
        self.subfigures.append([])

    def add_subfigure(self, subfigure: LatexSubfigure) -> None:
        """Adds a subfigure to the grid

        :param subfigure: The LatexSubfigure object
        :raises TypeError: Raises a TypeError if the subfigure is of the wrong type
        """
        if not isinstance(subfigure, LatexSubfigure):
            raise TypeError("Subfigure must be of type LatexSubfigure")
        self.subfigures[-1].append(subfigure)

    def _remove_empty_lists(self) -> None:
        """This will remove all the empty lists in the list of subfigure rows."""
        return_list = []
        for subfigs in self.subfigures:
            if len(subfigs) != 0:
                return_list.append(subfigs)
        self.subfigures = return_list

    @override
    def generate_latex_code(self, width: float = 0.99) -> str:
        """Generates the latex code for the subfigure grid

        :param width: The width of the subfigure grid, defaults to 0.99
        :return: The generated latex code
        """
        self._remove_empty_lists()

        latex_code = (
            f"{self._tab_prefix}\\begin{{{self._begin_type}}}[H]\n\\centering\n"
        )
        sublabel_counter = ord("a")
        for i, subfig_row in enumerate(self.subfigures):
            n_subfigs_in_row = len(subfig_row)
            if n_subfigs_in_row == 0:
                continue
            subfig_width = (width / n_subfigs_in_row) - 0.01
            for subfig in subfig_row:
                if subfig.label is None:
                    subfig.label = f"{self.label}_{chr(sublabel_counter)}"
                sublabel_counter += 1
                latex_code += subfig.generate_latex_code(width=subfig_width)
            if i < len(self.subfigures) - 1:
                latex_code += f"{self._tab_prefix}\\end{{{self._begin_type}}}\n\n"
                latex_code += f"{self._tab_prefix}\\begin{{{self._begin_type}}}[H]\n\\ContinuedFloat\n\\centering\n"

        if self.caption:
            latex_code += f"{self._tab_prefix}\\caption{{{self.caption}}}\n"
        if self.label:
            latex_code += (
                f"{self._tab_prefix}\\label{{{self._label_prefix}:{self.label}}}\n"
            )
        latex_code += f"{self._tab_prefix}\\end{{{self._begin_type}}}\n"
        return latex_code

    @override
    def save_figure(self, local_directory: str = LATEX_BASE_DIR) -> None:
        """Saves all the subfigures under their specified resoruce paths by calling their save_figure methods.

        :param local_directory: The local directory to save the figures under, defaults to LATEX_BASE_DIR
        """
        for subfig_row in self.subfigures:
            for subfig in subfig_row:
                subfig.save_figure(local_directory)
