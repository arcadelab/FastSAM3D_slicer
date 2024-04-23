from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from fastsamLib.fastsamLogic import fastsamLogic
import slicer
import os
import vtk
import qt
import urllib.request
import hashlib
import numpy as np


class fastsam(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "fastSAM"
        self.parent.categories = ["Segmentation"]
        self.parent.dependencies = []
        self.parent.contributors = [""]
        self.parent.helpText = \
            """TomoSAM helps with the segmentation of 3D data from tomography or other imaging techniques using the Segment Anything Model (SAM).<br>
<br>
General Tips (check out our <a href="https://github.com/fsemerar/SlicerTomoSAM">repository</a>):
<ul><li>Generate .pkl using <a href="https://colab.research.google.com/github/fsemerar/SlicerTomoSAM/blob/main/Embeddings/create_embeddings.ipynb">Colab</a> or the Embeddings Create button</li>
<li>Place .tif and .pkl in same folder and make their name equivalent</li>
<li>Drag and drop .tif --> imports both image and embeddings</li>
<li>Once include-point added, the slice is frozen (white background)</li>
<li>Accept Mask button clears points and confirms slice segmentation</li>
<li>Add Segment button adds a new label to segmentation</li>
<li>Create Interpolation button creates masks in between created ones</li>
<li>Undo button reverts interpolation or last mask</li>
<li>Clear button clears the points and the active mask</li>
</ul>
<br>
Note: SAM was trained with up to 9 points, so it is recommended to add up to 9 include+exclude points for optimal predictions<br>
<br>
Keyboard Shortcuts:
<ul><li>'i': switch to include-points</li>
<li>'e': switch to exclude-points</li>
<li>'a': accept mask</li>
<li>'n': new segment</li>
<li>'c': center view</li>
<li>'h': hide/show slice</li>
<li>'r': render 3D view</li>
<li>'z': undo interpolate or last mask</li>
</ul>"""
        self.parent.acknowledgementText = """If you find this work useful for your research or applications please cite <a href="https://arxiv.org/abs/2403.09827">our paper</a>"""


class fastsamWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False

        self.slice_visible = True
        self.actual_remove_click = True
        self.mask_accepted = False
        self.slice_frozen = False
        self.first_freeze = False

        self.lm = slicer.app.layoutManager()
        self.segmentEditorWidget = None
        self.layout_id = 20000
        self.orientation = 'horizontal'
        self.view = "Red"
        self.download_location = qt.QStandardPaths.writableLocation(qt.QStandardPaths.DownloadLocation)
        self.Fastsam3D_weights_path = os.path.join(self.download_location, "FastSAM3D.pth")
        self.SAMMed3D_weights_path = os.path.join(self.download_location,"SAM_Med3D_turbo.pth")
        self.MedSAM_weights_path = os.path.join(self.download_location, "medsam_vit_b.pth")
        self.SAMMed2D_weights_path = os.path.join(self.download_location, "sam-med2d_b.pth")
        self.layouts = {}  # Initialize an empty dictionary for layouts
        self.createLayouts()  # Call the method to create layouts

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/fastsam.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)
        self.logic = fastsamLogic()
        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.NodeAddedEvent, self.onNodeAdded)

        # volumes
        self.ui.comboVolumeNode.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.segmentSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.segmentSelector.connect("currentSegmentChanged(QString)", self.updateParameterNodeFromGUI)
        self.ui.segmentsTable.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)

        # markups
        self.ui.markupsInclude.connect("markupsNodeChanged()", self.updateParameterNodeFromGUI)
        self.ui.markupsExclude.connect("markupsNodeChanged()", self.updateParameterNodeFromGUI)
        self.ui.markupsInclude.markupsPlaceWidget().setPlaceModePersistency(True)
        self.ui.markupsExclude.markupsPlaceWidget().setPlaceModePersistency(True)

        # filepaths
        # self.ui.PathLineEdit_emb.connect('currentPathChanged(QString)', self.pathEmbeddings)

        # push buttons
        self.ui.pushSAMweights.connect("clicked(bool)", self.onPushSAMweights)
        self.ui.pushSAMweights.setToolTip(f"Download SAM weights to {self.Fastsam3D_weights_path}")
        self.ui.pushMaskAccept.connect("clicked(bool)", self.onPushMaskAccept)
        self.ui.pushMaskClear.connect("clicked(bool)", self.onPushMaskClear)
        self.ui.pushSegmentAdd.connect("clicked(bool)", self.onPushSegmentAdd)
        self.ui.pushSegmentRemove.connect("clicked(bool)", self.onPushSegmentRemove)
        # self.ui.pushCenter3d.connect("clicked(bool)", self.onPushCenter3d)
        # self.ui.pushVisualizeSlice3d.connect("clicked(bool)", self.onPushVisualizeSlice3d)
        self.ui.pushInitializeInterp.connect("clicked(bool)", self.onPushInitializeInterp)
        self.ui.pushUndo.connect("clicked(bool)", self.onPushUndo)
        self.ui.radioButton_FastSAM3D.connect("clicked(bool)", self.buildFastSAM3D)
        self.ui.radioButton_SAMMed3D.connect("clicked(bool)", self.buildSAMMed3D)
        self.ui.radioButton_SAMMed2D.connect("clicked(bool)", self.buildSAMMed2D)
        self.ui.radioButton_MedSAM.connect("clicked(bool)", self.buildMedSAM)

        shortcuts = [
            ("i", lambda: self.activateIncludePoints()),
            ("e", lambda: self.activateExcludePoints()),
            ("a", lambda: self.onPushMaskAccept()),
            ("n", lambda: self.onPushSegmentAdd()),
            ("c", lambda: self.onPushCenter3d()),
            ("h", lambda: self.onPushVisualizeSlice3d()),
            ("r", lambda: self.onPushRender3d()),
            ("z", lambda: self.onPushUndo()),
        ]
        for (shortcutKey, callback) in shortcuts:
            shortcut = qt.QShortcut(qt.QKeySequence(shortcutKey), slicer.util.mainWindow())
            shortcut.connect("activated()", callback)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

        # Attach to logic
        self.logic.ui = self.ui

    def initializeParameterNode(self):
        """
        Ensure parameter node exists and observed.
        """
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.
        self.setParameterNode(self.logic.getParameterNode())

        self._parameterNode.root_path = os.path.dirname(os.path.dirname(os.path.dirname(self.resourcePath(''))))
        if self.logic.sam is None:
            self.logic.create_sam(self.Fastsam3D_weights_path, "FastSAM3D")
            # self.updateLayout()

        if self._parameterNode.GetNodeReferenceID("fastsamInputVolume") is None:
            volume_node = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if volume_node is not None:
                self._parameterNode.SetNodeReferenceID("fastsamInputVolume", volume_node.GetID())
                self.importPKL(volume_node)

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.GetNodeReferenceID("fastsamIncludePoints"):
            markupsNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "include-points")
            self._parameterNode.SetNodeReferenceID("fastsamIncludePoints", markupsNode.GetID())
            markupsNode.GetDisplayNode().SetSelectedColor(0, 1, 0)
            markupsNode.GetDisplayNode().SetActiveColor(0, 1, 0)
            markupsNode.GetDisplayNode().SetTextScale(0)
            markupsNode.GetDisplayNode().SetGlyphScale(1)
            markupsNode.AddObserver(slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent, self.onMarkupIncludePointPositionDefined)
            markupsNode.AddObserver(slicer.vtkMRMLMarkupsNode.PointPositionUndefinedEvent, self.onMarkupIncludePointPositionUndefined)

        if not self._parameterNode.GetNodeReferenceID("fastsamExcludePoints"):
            markupsNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "exclude-points")
            self._parameterNode.SetNodeReferenceID("fastsamExcludePoints", markupsNode.GetID())
            markupsNode.GetDisplayNode().SetSelectedColor(1, 0, 0)
            markupsNode.GetDisplayNode().SetActiveColor(1, 0, 0)
            markupsNode.GetDisplayNode().SetTextScale(0)
            markupsNode.GetDisplayNode().SetGlyphScale(1)
            markupsNode.AddObserver(slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent, self.onMarkupExcludePointPositionDefined)
            markupsNode.AddObserver(slicer.vtkMRMLMarkupsNode.PointPositionUndefinedEvent, self.onMarkupExcludePointPositionUndefined)

        if not self._parameterNode.GetNodeReferenceID("fastsamSegmentation"):
            segmentationNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode', 'Segmentation')
            self.ui.segmentSelector.setCurrentNode(segmentationNode)
            self.ui.segmentsTable.setSegmentationNode(segmentationNode)
            self._parameterNode.SetNodeReferenceID("fastsamSegmentation", segmentationNode.GetID())
            segmentationNode.CreateDefaultDisplayNodes()
            segmentID = segmentationNode.GetSegmentation().AddEmptySegment()
            self._parameterNode.SetParameter("fastsamCurrentSegment", segmentID)

        if self.segmentEditorWidget is None:
            self.segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
            self.segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
            self.segmentEditorNode = slicer.vtkMRMLSegmentEditorNode()
            slicer.mrmlScene.AddNode(self.segmentEditorNode)
            self.segmentEditorWidget.setMRMLSegmentEditorNode(self.segmentEditorNode)

            self.segmentEditorWidget.setSegmentationNode(self._parameterNode.GetNodeReference("fastsamSegmentation"))
            self.segmentEditorWidget.setSourceVolumeNode(self._parameterNode.GetNodeReference("fastsamInputVolume"))
            self.segmentEditorWidget.setActiveEffectByName("Fill between slices")
            self.effect = self.segmentEditorWidget.activeEffect()

    def updateGUIFromParameterNode(self, caller=None, event=None):
        """
        This method is called whenever parameter node is changed.
        The module GUI is updated to show the current state of the parameter node.
        """
        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
        self._updatingGUIFromParameterNode = True

        # Update node selectors and sliders
        self.ui.comboVolumeNode.setCurrentNode(self._parameterNode.GetNodeReference("fastsamInputVolume"))

        self.ui.markupsInclude.setCurrentNode(self._parameterNode.GetNodeReference("fastsamIncludePoints"))
        self.ui.markupsExclude.setCurrentNode(self._parameterNode.GetNodeReference("fastsamExcludePoints"))

        if self._parameterNode.GetNodeReferenceID("fastsamSegmentation"):
            self._parameterNode.GetNodeReference("fastsamSegmentation").SetReferenceImageGeometryParameterFromVolumeNode(
                self._parameterNode.GetNodeReference("fastsamInputVolume"))
            self.ui.segmentsTable.setSegmentationNode(self._parameterNode.GetNodeReference("fastsamSegmentation"))

        if self.segmentEditorWidget:
            self.segmentEditorWidget.setSegmentationNode(self._parameterNode.GetNodeReference("fastsamSegmentation"))
            self.segmentEditorWidget.setSourceVolumeNode(self._parameterNode.GetNodeReference("fastsamInputVolume"))

        # All the GUI updates are done
        self._updatingGUIFromParameterNode = False

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        """
        This method is called when the user makes any change in the GUI.
        The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
        """
        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

        if self.ui.comboVolumeNode.currentNodeID != self._parameterNode.GetNodeReferenceID("fastsamInputVolume"):
            self._parameterNode.SetNodeReferenceID("fastsamInputVolume", self.ui.comboVolumeNode.currentNodeID)
            self.importPKL(self._parameterNode.GetNodeReference("fastsamInputVolume"))

        if self._parameterNode.GetNodeReference("fastsamInputVolume") is not None:
            slicer.util.setSliceViewerLayers(background=self._parameterNode.GetNodeReference("fastsamInputVolume"))

        self._parameterNode.SetNodeReferenceID("fastsamIncludePoints", self.ui.markupsInclude.currentNode().GetID())
        self._parameterNode.SetNodeReferenceID("fastsamExcludePoints", self.ui.markupsExclude.currentNode().GetID())

        self._parameterNode.SetNodeReferenceID("fastsamSegmentation", self.ui.segmentSelector.currentNodeID())
        self._parameterNode.GetNodeReference("fastsamSegmentation").SetReferenceImageGeometryParameterFromVolumeNode(
            self._parameterNode.GetNodeReference("fastsamInputVolume"))
        self._parameterNode.SetParameter("fastsamCurrentSegment", self.ui.segmentSelector.currentSegmentID())
        self.ui.segmentsTable.setSegmentationNode(self._parameterNode.GetNodeReference("fastsamSegmentation"))

        if self.segmentEditorWidget:
            self.segmentEditorWidget.setSegmentationNode(self._parameterNode.GetNodeReference("fastsamSegmentation"))
            self.segmentEditorWidget.setSourceVolumeNode(self._parameterNode.GetNodeReference("fastsamInputVolume"))

        self._parameterNode.EndModify(wasModified)

    def findClickedSliceWindow(self):
        sliceNode = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLCrosshairNode').GetCursorPositionXYZ([0, 0, 0])
        if sliceNode is None:
            return
        return self.lm.sliceWidget(sliceNode.GetName()).sliceLogic().GetSliceNode().GetLayoutName()

    # to avoid repetitions, calling the same pointAdded/pointRemoved functions
    def onMarkupIncludePointPositionDefined(self, caller, event):

        slice_dir = self.findClickedSliceWindow()
        # if slice_dir is None:
        #     self.actual_remove_click = False
        #     self.mask_accepted = True
        #     caller.RemoveNthControlPoint(caller.GetDisplayNode().GetActiveControlPoint())
        #     return

        if caller.GetNumberOfControlPoints() == 1 and not self.slice_frozen:
            self.logic.slice_direction = slice_dir
            # self.onPushVisualizeSlice3d(not_from_button=True)
            self.freezeSlice()
            self.first_freeze = True

        # if self.logic.slice_direction != slice_dir:
        #     self.actual_remove_click = False
        #     caller.RemoveNthControlPoint(caller.GetDisplayNode().GetActiveControlPoint())
        #     return

        self.addPoint(caller, self.logic.include_coords)
        self.first_freeze = False

    def onMarkupIncludePointPositionUndefined(self, caller, event):

        if caller.GetNumberOfControlPoints() == 0:
            self.unfreezeSlice()
            self.actual_remove_click = False
            self._parameterNode.GetNodeReference("fastsamExcludePoints").RemoveAllControlPoints()
            self.actual_remove_click = True
            self.logic.exclude_coords = {}

        if self.mask_accepted:
            self.mask_accepted = False
            return

        self.removePoint(caller, self.logic.include_coords, 'include')

    def onMarkupExcludePointPositionDefined(self, caller, event):

        slice_dir = self.findClickedSliceWindow()
        if (self._parameterNode.GetNodeReference("fastsamIncludePoints").GetNumberOfControlPoints() == 0 or
                slice_dir is None or self.logic.slice_direction != slice_dir):
            self.actual_remove_click = False
            caller.RemoveNthControlPoint(caller.GetDisplayNode().GetActiveControlPoint())
            return

        self.addPoint(caller, self.logic.exclude_coords)

    def onMarkupExcludePointPositionUndefined(self, caller, event):
        self.removePoint(caller, self.logic.exclude_coords, 'exclude')

    def addPoint(self, caller, stored_coords):
        self.logic.generateaffine()
        point_index = caller.GetDisplayNode().GetActiveControlPoint()
        
        if not self.checkVolume() or not self.checkSAM():
            self.actual_remove_click = False
            caller.RemoveNthControlPoint(point_index)
            return
        coords = caller.GetNthControlPointPosition(point_index)
        
        coords = np.array([coords[0],coords[1],coords[2],1])
        coords = np.round(np.linalg.inv(self.logic.affine) @ np.array(coords).T)
        caller.SetNthControlPointLocked(point_index, True)
        coords = np.array([coords[2],coords[1],coords[0]])
        # print(coords)
        # coords = (int(round(coords[2])), int(round(coords[1])), int(round(coords[0])))
        inside_slice_check = True
        for coord in coords:
            if coord < 0:
                inside_slice_check = False
        if not (coords[0]<self.logic.img.shape[0] and coords[1] < self.logic.img.shape[1] and coords[2] < self.logic.img.shape[2]):
            inside_slice_check = False
        # if self.logic.slice_direction == 'Red':
        #     if not (coords[2] < self.logic.img.shape[2] and coords[1] < self.logic.img.shape[1]):
        #         inside_slice_check = False
        # elif self.logic.slice_direction == 'Green':
        #     if not (coords[2] < self.logic.img.shape[2] and coords[0] < self.logic.img.shape[0]):
        #         inside_slice_check = False
        # else:
        #     if not (coords[1] < self.logic.img.shape[1] and coords[0] < self.logic.img.shape[0]):
        #         inside_slice_check = False
        # print(coords)
        if not inside_slice_check:
            self.actual_remove_click = False
            caller.RemoveNthControlPoint(point_index)
            return
        stored_coords[caller.GetNthControlPointLabel(point_index)] = coords
        self.logic.get_mask(self.first_freeze)

    def removePoint(self, caller, stored_coords, operation):

        # this is to avoid calling remove internally when an illegal point has been added
        # i.e. only call it when user actually removes a point
        if not self.actual_remove_click:
            self.actual_remove_click = True
            return

        new_coords = {}
        for i in range(caller.GetNumberOfControlPoints()):
            point_label = caller.GetNthControlPointLabel(i)
            new_coords[point_label] = stored_coords[point_label]
        if operation == 'include':
            self.logic.include_coords = new_coords
        else:
            self.logic.exclude_coords = new_coords
        self.logic.get_mask(self.first_freeze)

    def freezeSlice(self):
        slice_widget = self.lm.sliceWidget(self.logic.slice_direction)
        interactorStyle = slice_widget.sliceView().sliceViewInteractorStyle()
        interactorStyle.SetActionEnabled(interactorStyle.BrowseSlice, False)
        slice_widget.sliceView().setBackgroundColor(qt.QColor.fromRgbF(1, 1, 1))
        self.slice_frozen = True
        slice_widget.sliceController().setDisabled(self.slice_frozen)  # freeze slidebar

    def unfreezeSlice(self):
        slice_widget = self.lm.sliceWidget(self.logic.slice_direction)
        interactorStyle = slice_widget.sliceView().sliceViewInteractorStyle()
        interactorStyle.SetActionEnabled(interactorStyle.BrowseSlice, True)
        slice_widget.sliceView().setBackgroundColor(qt.QColor.fromRgbF(0, 0, 0))
        self.slice_frozen = False
        slice_widget.sliceController().setDisabled(self.slice_frozen)  # freeze slidebar

    def clearPoints(self):
        self.actual_remove_click = False
        self._parameterNode.GetNodeReference("fastsamIncludePoints").RemoveAllControlPoints()
        self.logic.include_coords = {}
        self.actual_remove_click = False
        self._parameterNode.GetNodeReference("fastsamExcludePoints").RemoveAllControlPoints()
        self.logic.exclude_coords = {}

    def onPushMaskAccept(self):
        self.mask_accepted = True
        self.actual_remove_click = False
        self._parameterNode.GetNodeReference("fastsamIncludePoints").RemoveAllControlPoints()
        self.logic.include_coords = {}

        # only add mask_location for interpolation if mask has been created
        if ((self.logic.slice_direction == 'Red' and self.logic.mask[self.logic.ind].max() != 0) or
                (self.logic.slice_direction == 'Green' and self.logic.mask[:, self.logic.ind].max() != 0) or
                (self.logic.slice_direction == 'Yellow' and self.logic.mask[:, :, self.logic.ind].max() != 0)):
            self.logic.mask_locations.add(self.logic.ind)
            self.logic.interp_slice_direction.add(self.logic.slice_direction)

    def onPushMaskClear(self):
        if self.slice_frozen:
            self.logic.undo()
            self.onPushMaskAccept()

    def onPushUndo(self):
        self.logic.undo()

    def onPushSegmentAdd(self):
        self.clearPoints()
        self.logic.mask_locations = set()
        self.logic.interp_slice_direction = set()
        segmentID = self._parameterNode.GetNodeReference("fastsamSegmentation").GetSegmentation().AddEmptySegment()
        self._parameterNode.SetParameter("fastsamCurrentSegment", segmentID)
        self.ui.segmentSelector.setCurrentSegmentID(segmentID)
        self.logic.mask.fill(0)

    def onPushSegmentRemove(self):
        if len(self._parameterNode.GetNodeReference("fastsamSegmentation").GetSegmentation().GetSegmentIDs()) <= 1:
            slicer.util.errorDisplay("Need to have at least one segment")
            return
        self._parameterNode.GetNodeReference("fastsamSegmentation").RemoveSegment(self._parameterNode.GetParameter("fastsamCurrentSegment"))
        self.ui.segmentSelector.setCurrentSegmentID(self._parameterNode.GetNodeReference("fastsamSegmentation").GetSegmentation().GetSegmentIDs()[-1])

    def onPushSAMweights(self):
        if self.checkFastSAMdownload():
            slicer.util.delayDisplay(f"Downloading FastSAM3D weights to {self.Fastsam3D_weights_path}(200M, it may take several minutes)...")
            print("Downloading FastSAM3D weights ... ", end='')
            url = "https://www.googleapis.com/drive/v3/files/1gnjBRyX6-09OjKispBlt2l-Jpe62Skk2?alt=media&key=AIzaSyDIQ5koL_iOuj5vIxaRKtUwba70-DH5nTc"
            urllib.request.urlretrieve(url, self.Fastsam3D_weights_path)
            print("Done")
        else:
            slicer.util.infoDisplay(f"FastSAM3D weights already found at {self.Fastsam3D_weights_path}(1.1G, it may take several minutes)...")
        
        if self.checkSAMMeddownload():
            slicer.util.delayDisplay(f"Downloading SAMMed3D weights to {self.SAMMed3D_weights_path}")
            print("Downloading SAMMed3D weights ... ", end='')
            url = "https://www.googleapis.com/drive/v3/files/1MuqYRQKIZb4YPtEraK8zTKKpp-dUQIR9?alt=media&key=AIzaSyDIQ5koL_iOuj5vIxaRKtUwba70-DH5nTc"
            urllib.request.urlretrieve(url, self.SAMMed3D_weights_path)
            print("Done")
        else:
            slicer.util.infoDisplay(f"MedSAM3D weights already found at {self.SAMMed3D_weights_path}")
        
        if self.checkMedSAMdownload():
            slicer.util.delayDisplay(f"Downloading MedSAM weights to {self.MedSAM_weights_path}")
            print("Downloading MedSAM weights ... ", end='')
            url = "https://www.googleapis.com/drive/v3/files/1UAmWL88roYR7wKlnApw5Bcuzf2iQgk6_?alt=media&key=AIzaSyDIQ5koL_iOuj5vIxaRKtUwba70-DH5nTc"
            urllib.request.urlretrieve(url, self.MedSAM_weights_path)
            print("Done")
        else:
            slicer.util.infoDisplay(f"MedSAM3D weights already found at {self.MedSAM_weights_path}")
        
        if self.checkSAMMed2Ddownload():
            slicer.util.delayDisplay(f"Downloading MedSAM weights to {self.SAMMed2D_weights_path}")
            print("Downloading MedSAM weights ... ", end='')
            url = "https://www.googleapis.com/drive/v3/files/1ARiB5RkSsWmAB_8mqWnwDF8ZKTtFwsjl?alt=media&key=AIzaSyDIQ5koL_iOuj5vIxaRKtUwba70-DH5nTc"
            urllib.request.urlretrieve(url, self.SAMMed2D_weights_path)
            print("Done")
        else:
            slicer.util.infoDisplay(f"MedSAM3D weights already found at {self.SAMMed2D_weights_path}")
            

    def checkFastSAMdownload(self):
        return not os.path.exists(self.Fastsam3D_weights_path)  or not os.path.isfile(self.Fastsam3D_weights_path) 
    
    def checkSAMMeddownload(self):
        return not os.path.exists(self.SAMMed3D_weights_path) or not os.path.isfile(self.SAMMed3D_weights_path)
    
    def checkMedSAMdownload(self):
        return not os.path.exists(self.MedSAM_weights_path) or not os.path.isfile(self.MedSAM_weights_path)
    
    def checkSAMMed2Ddownload(self):
        return not os.path.exists(self.SAMMed2D_weights_path) or not os.path.isfile(self.SAMMed2D_weights_path)
    
    def buildFastSAM3D(self):
        self.logic.dimension = 3
        self.logic.image_size = 128
        self.logic.create_sam(self.Fastsam3D_weights_path, "FastSAM3D")
    
    def buildSAMMed3D(self):
        self.logic.dimension = 3
        self.logic.image_size = 128
        self.logic.create_sam(self.SAMMed3D_weights_path, "SAMMed3D")
        
    def buildSAMMed2D(self):
        self.logic.dimension = 2
        self.logic.image_size = 256
        self.logic.create_sam(self.SAMMed2D_weights_path, "SAMMed2D")
    
    def buildMedSAM(self):
        self.logic.dimension = 2
        self.logic.image_size = 1024
        self.logic.create_sam(self.MedSAM_weights_path, "MedSAM")
    
    def checkSAM(self):
        if self.logic.sam is None:
            if self.checkFastSAMdownload():
                slicer.util.errorDisplay("SAM weights not found, use Download button")
                return False
            self.logic.create_sam(self.Fastsam3D_weights_path,'vit_b_ori')
        return True

    # def checkEmbeddings(self):
    #     # if len(self.logic.embeddings) == 0:
    #     #     slicer.util.errorDisplay("Select image Embeddings")
    #     #     return False
    #     return True

    def checkVolume(self):
        if not self._parameterNode.GetNodeReferenceID("fastsamInputVolume"):
            slicer.util.errorDisplay("Select a volume")
            return False
        else:
            return True

    # def pathEmbeddings(self):
    #     embeddings_path = self.ui.PathLineEdit_emb.currentPath

    #     if not os.path.exists(embeddings_path) or not os.path.isfile(embeddings_path):
    #         slicer.util.errorDisplay("Image Embeddings file not found")
    #         return
    #     elif os.path.splitext(embeddings_path)[1] != ".pkl":
    #         slicer.util.errorDisplay("Unrecognized extension for image Embeddings")
    #         return

    #     slicer.util.delayDisplay(f"Reading embeddings: {embeddings_path}")
    #     self.logic.read_img_embeddings(embeddings_path)

    # def onPushEmbeddingsCreate(self):
    #     if not self.checkVolume() or not self.checkSAM():
    #         return

    #     volume_node = self._parameterNode.GetNodeReference("fastsamInputVolume")
    #     storageNode = volume_node.GetStorageNode()
    #     if storageNode is not None:  # loaded via drag-drop
    #         filepath = storageNode.GetFullNameFromFileName()
    #     else:  # Loaded via DICOM browser
    #         instanceUIDs = volume_node.GetAttribute("DICOM.instanceUIDs").split()
    #         filepath = slicer.dicomDatabase.fileForInstance(instanceUIDs[0])

    #     output_filepath = os.path.join(self.download_location, os.path.splitext(filepath)[0] + ".pkl")
    #     if os.path.isfile(output_filepath):
    #         slicer.util.infoDisplay(f"Embeddings file already found at {output_filepath}")
    #         return

    #     if slicer.util.confirmOkCancelDisplay("It is recommended to create embeddings on a machine with GPU (e.g. Colab) to reduce runtime.\n"
    #                                           "Creating them locally may take several minutes and there is not way to stop the process other than force quitting Slicer.\n"
    #                                           "Click OK to continue"):
    #         self.embeddings_already_loaded = True
    #         self.ui.PathLineEdit_emb.currentPath = self.logic.create_embeddings(output_filepath)

    # def onPushEmbeddingsColab(self):
    #     qt.QDesktopServices.openUrl(qt.QUrl("https://colab.research.google.com/github/fsemerar/SlicerTomoSAM/blob/main/Embeddings/create_embeddings.ipynb"))

    # def onPushCenter3d(self):
    #     if not self.slice_frozen:
    #         threeDWidget = self.lm.threeDWidget(0)
    #         threeDView = threeDWidget.threeDView()
    #         threeDView.resetFocalPoint()
    #         slicer.util.resetSliceViews()

    # def onPushVisualizeSlice3d(self, not_from_button=False):
    #     if not not_from_button:
    #         self.slice_visible = not self.slice_visible
    #     # first deactivate them all
    #     for sliceViewName in self.lm.sliceViewNames():
    #         self.lm.sliceWidget(sliceViewName).sliceController().setSliceVisible(False)
    #     # then activate only the one we need
    #     self.lm.sliceWidget(self.logic.slice_direction).sliceController().setSliceVisible(self.slice_visible)

    # def onPushRender3d(self):
    #     self._parameterNode.GetNodeReference("fastsamSegmentation").CreateClosedSurfaceRepresentation()

    def onPushInitializeInterp(self):
        if len(self.logic.interp_slice_direction) > 1:
            slicer.util.errorDisplay("Cannot interpolate if multiple slice directions have been segmented")
            return
        elif len(self.logic.interp_slice_direction) == 0 or len(self.logic.mask_locations) == 0:
            slicer.util.errorDisplay("Cannot interpolate if no masks have been added")
            return

        self.logic.backup_mask()

        # when segments not visible, interpolation won't happen --> hide everything except current segment
        for segmentID in self._parameterNode.GetNodeReference("fastsamSegmentation").GetSegmentation().GetSegmentIDs():
            if segmentID != self._parameterNode.GetParameter("fastsamCurrentSegment"):
                self._parameterNode.GetNodeReference("fastsamSegmentation").GetDisplayNode().SetSegmentVisibility(segmentID, False)

        self.effect.self().onPreview()
        self.effect.self().onApply()

        for segmentID in self._parameterNode.GetNodeReference("fastsamSegmentation").GetSegmentation().GetSegmentIDs():
            if segmentID != self._parameterNode.GetParameter("fastsamCurrentSegment"):
                self._parameterNode.GetNodeReference("fastsamSegmentation").GetDisplayNode().SetSegmentVisibility(segmentID, True)

    @vtk.calldata_type(vtk.VTK_OBJECT)
    def onNodeAdded(self, caller, event, calldata):
        node = calldata
        if type(node) == slicer.vtkMRMLScalarVolumeNode:
            # call using a timer instead of calling it directly to allow the volume loading to fully complete
            print("Reading image ... ", end='')
            qt.QTimer.singleShot(30, lambda: self.autoSelectVolume(node))
            print("Done")
            qt.QTimer.singleShot(30, lambda: self.importPKL(node))

    def autoSelectVolume(self, volumeNode):
        self.ui.comboVolumeNode.setCurrentNodeID(volumeNode.GetID())

    def importPKL(self, volumeNode):
        if volumeNode is not None:
            storage_node = volumeNode.GetStorageNode()
            if storage_node is not None:
                pkl_filepath = os.path.splitext(storage_node.GetFileName())[0] + ".pkl"
                if os.path.exists(pkl_filepath):  # try to import Embeddings too if same name and folder as image
                    self.ui.PathLineEdit_emb.setCurrentPath(pkl_filepath)
                # self.onPushCenter3d()  # also run these functions on import
                # self.onPushVisualizeSlice3d(not_from_button=True)

    def activateIncludePoints(self):
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        selectionNode = slicer.app.applicationLogic().GetSelectionNode()
        selectionNode.SetActivePlaceNodeID(self._parameterNode.GetNodeReferenceID("fastsamIncludePoints"))
        interactionNode.SetCurrentInteractionMode(interactionNode.Place)

    def activateExcludePoints(self):
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        selectionNode = slicer.app.applicationLogic().GetSelectionNode()
        selectionNode.SetActivePlaceNodeID(self._parameterNode.GetNodeReferenceID("tomosamExcludePoints"))
        interactionNode.SetCurrentInteractionMode(interactionNode.Place)

    # def onRadioOrient(self):
    #     if self.ui.radioButton_hor.isChecked():
    #         self.orientation = 'horizontal'
    #     else:
    #         self.orientation = 'vertical'
    #     self.updateLayout()

    # def onRadioView(self):
    #     if self.ui.radioButton_red.isChecked():
    #         self.view = 'Red'
    #     elif self.ui.radioButton_green.isChecked():
    #         self.view = 'Green'
    #     else:
    #         self.view = 'Yellow'
    #     self.updateLayout()

    def createLayouts(self):
        orientations = ['horizontal', 'vertical']
        views = ['Red', 'Green', 'Yellow']

        for orientation in orientations:
            for view in views:
                self.layout_id += 1
                customLayout = f"""
                    <layout type="{orientation}" split="true">
                        <item>
                            <view class="vtkMRMLViewNode" singletontag="1">
                                <property name="viewlabel" action="default">1</property>
                            </view>
                        </item>
                        <item>
                            <view class="vtkMRMLSliceNode" singletontag="{view}">
                            </view>
                        </item>
                    </layout>
                """
                self.layouts[(orientation, view)] = self.layout_id
                self.lm.layoutLogic().GetLayoutNode().AddLayoutDescription(self.layout_id, customLayout)
                self.layout_id += 1

    # def updateLayout(self):
    #     self.layout_id = self.layouts[(self.orientation, self.view)]

    #     # If the layout ID doesn't exist, use a default layout
    #     if self.layout_id is None:
    #         defaultLayout = self.layouts[('horizontal', 'Red')]
    #         if defaultLayout is None:
    #             return
    #         self.layout_id = defaultLayout

        # self.lm.setLayout(self.layout_id)
        # self.logic.slice_direction = self.view
        # self.onPushVisualizeSlice3d(not_from_button=True)

    # methods below don't need to be changed
    def cleanup(self):
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()

    def enter(self):
        """
        Called each time the user opens this module.
        """
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self):
        """
        Called each time the user opens a different module.
        """
        # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
        self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    def onSceneStartClose(self, caller, event):
        """
        Called just before the scene is closed.
        """
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event):
        """
        Called just after the scene is closed.
        """
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def setParameterNode(self, inputParameterNode):
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """
        # Unobserve previously selected parameter node and add an observer to the newly selected.
        # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
        # those are reflected immediately in the GUI.
        if self._parameterNode is not None:
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self._parameterNode = inputParameterNode
        if self._parameterNode is not None:
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

        # Initial GUI update
        self.updateGUIFromParameterNode()
