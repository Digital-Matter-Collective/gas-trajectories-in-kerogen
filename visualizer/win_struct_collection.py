from vtkmodules.vtkRenderingCore import (
    vtkRenderWindowInteractor,
)

from visualizer.interactor_styles import KeyPressInteractorStyle


class WinStructCollection:
    def __init__(self, interactor: vtkRenderWindowInteractor):
        self.interactor = interactor
        self.renWin = self.interactor.GetRenderWindow()
        self.kpis = KeyPressInteractorStyle(iren=self.interactor)
        self.interactor.SetInteractorStyle(self.kpis)
        self.running = True

    def clear(self) -> None:
        self.renWin.Finalize()
        del self.renWin, self.interactor
