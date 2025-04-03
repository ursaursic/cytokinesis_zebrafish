# adapted from https://imagej.net/plugins/trackmate/scripting/scripting, modified by Ursa Ursic

import sys
import os
from ij import IJ
from ij import WindowManager
import ij.io.FileSaver as FileSaver

from fiji.plugin.trackmate import Logger, Model, Settings, TrackMate, SelectionModel

from fiji.plugin.trackmate.io import TmXmlWriter
from fiji.plugin.trackmate.util import TMUtils, LogRecorder
from fiji.plugin.trackmate.action import CaptureOverlayAction
from fiji.plugin.trackmate.detection import DogDetectorFactory
from fiji.plugin.trackmate.tracking.jaqaman import LAPUtils, SparseLAPTrackerFactory 
from fiji.plugin.trackmate.gui.displaysettings import DisplaySettingsIO, DisplaySettings 
from fiji.plugin.trackmate.visualization.hyperstack import HyperStackDisplayer

import java.io.File as File
import fiji.plugin.trackmate.features.FeatureFilter as FeatureFilter

import csv

reload(sys)
sys.setdefaultencoding('utf-8')
show_output = True


# Specify the path to your CSV file
csv_file_path = "/Volumes/ursic/Projects/extract_material_properties/calibration_magnetic_tweezers/1_analyses/calibration_files_overview.csv"

files = []
track_files = []

if os.path.exists(csv_file_path):
	# Open the CSV file
	with open(csv_file_path, 'r') as file:
		# Create a CSV reader object
		csv_reader = csv.reader(file)
	
		# Iterate over each row in the CSV file
		for row in csv_reader:
			files.append(row[0].split(';')[0])
			track_files.append(row[0].split(';')[1])
			
			print(row)

else:
	csv_file_path = csv_file_path.replace('/Volumes/', '\\\\fileserver.mpi-cbg.de\\').replace('/', '\\')
	
	# Open the CSV file
	with open(csv_file_path, 'r') as file:
		# Create a CSV reader object
		csv_reader = csv.reader(file)
	
		# Iterate over each row in the CSV file
		for row in csv_reader:
			files.append(row[0].split(';')[0].replace(';', '').replace('/Volumes/', '\\\\fileserver.mpi-cbg.de\\').replace('/', '\\'))
			track_files.append(row[0].split(';')[1].replace(';', '').replace('/Volumes/', '\\\\fileserver.mpi-cbg.de\\').replace('/', '\\'))

files.pop(0)

for imp_path in files:
	print(imp_path)
	# Get the image
	imp = IJ.openImage(imp_path)
	imp.show()

	if imp is not None:
		# Get the dimensions of the image
		n_slices = imp.getNSlices()
		n_frames = imp.getNFrames()
		n_channels = imp.getNChannels()
	
	pix_size = 1.083333

	imp.getCalibration().setXUnit("um")
	IJ.run(imp, "Properties...", "channels="+ str(n_channels) + " slices=" + str(n_slices) + " frames=" + str(n_frames) +  " pixel_width=" + str(pix_size) + " pixel_height="  + str(pix_size) + " voxel_depth=" + str(pix_size))
	
	if n_slices > n_frames:
		# Reorder hyperstack
		IJ.run(imp, "Re-order Hyperstack ...", "channels=[Channels (c)] slices=[Frames (t)] frames=[Slices (z)]")
		
		# Check if the reordering was successful
		if imp is None:
		    print("Error: Failed to reorder hyperstack.")
		    sys.exit(1)

	
	# Save the reordered hyperstack to a file
	# IJ.saveAs(imp, "Tiff", imp_path)
	FileSaver(IJ.getImage()).saveAsTiff(imp_path)
	print("TIFF file saved successfully.")
	
	imp.close()
	
	# Reopen the reordered hyperstack
	imp = IJ.openImage(imp_path)
		
	# Logger -> content will be saved in the XML file.
	logger = LogRecorder(Logger.VOID_LOGGER )
	
	
	#-------------------------
	# Instantiate model object
	#-------------------------
	model = Model()
	
	
	#------------------------
	# Prepare settings object
	#------------------------``
	settings = Settings(imp)
	
	# Configure detector
	settings.detectorFactory = DogDetectorFactory()
	settings.detectorSettings = {
	    'DO_SUBPIXEL_LOCALIZATION' : True,
	    'RADIUS' : 2.5,
	    'TARGET_CHANNEL' : 1,
	    'THRESHOLD' : 175.0,
	    'DO_MEDIAN_FILTERING' : True
	}
	
	# Configure tracker
	settings.trackerFactory = SparseLAPTrackerFactory()
	settings.trackerSettings = settings.trackerFactory.getDefaultSettings()
	
	settings.trackerSettings['LINKING_MAX_DISTANCE'] = 15.0
	settings.trackerSettings['GAP_CLOSING_MAX_DISTANCE'] = 15.0
	settings.trackerSettings['MAX_FRAME_GAP'] = 2
	
	# Add the analyzers for some spot features.
	# Here we decide brutally to add all of them.
	settings.addAllAnalyzers()
	
	#filter2 = FeatureFilter('TOTAL_DISTANCE_TRAVELED', 13.0, True)
	#settings.addTrackFilter(filter2)
	
	# We configure the initial filtering to discard spots 
	# with a quality lower than 1.
	#settings.initialSpotFilterValue = 1.
	
	
	
	#----------------------
	# Instantiate trackmate
	#----------------------
	
	trackmate = TrackMate(model, settings)
	
	#------------
	# Execute all
	#------------
	
	
	ok = trackmate.checkInput()
	if not ok:
	    sys.exit(str(trackmate.getErrorMessage()))
	
	ok = trackmate.process()
	if not ok:
	    sys.exit(str(trackmate.getErrorMessage()))
	
	fm = model.getFeatureModel()
	logger.log('TRACK_ID' + '\t' + 'FRAME' + '\t' + 'POSITION_X' + '\t' + 'POSITION_Y' + '\t' + 'POSITION_T' + '\t' + 'QUALITY' + '\t'  + 'SNR_CH1' + '\t' + 'MEAN_INTENSITY_CH1' + '\n')
	
	# Iterate over all the tracks that are visible.
	for id in model.getTrackModel().trackIDs(True):
	
	    v = fm.getTrackFeature(id, 'TRACK_MEAN_SPEED')
	
	    # Get all the spots of the current track.
	    track = model.getTrackModel().trackSpots(id)
	    for spot in track:
	        sid = spot.ID()
	        # Fetch spot features directly from spot.
	        # Note that for spots the feature values are not stored in the FeatureModel
	        # object, but in the Spot object directly. This is an exception; for tracks
	        # and edges, you have to query the feature model.
	        x = spot.getFeature('POSITION_X')
	        y = spot.getFeature('POSITION_Y')
	        f = spot.getFeature('FRAME')
	        t = spot.getFeature('POSITION_T')
	        q = spot.getFeature('QUALITY')
	        snr = spot.getFeature('SNR_CH1')
	        mean = spot.getFeature('MEAN_INTENSITY_CH1')
	        logger.log(str(id) + '\t' + str(f) + '\t' + str(x) + '\t' + str(y) + '\t'+ str(t)+ '\t' + str(q) + '\t' +  str(snr) +'\t' + str(mean)+ '\n')
	
	
	# ----------------
	# Save results
	# ----------------
	
	saveFile = TMUtils.proposeTrackMateSaveFile(settings, logger)
	
	writer = TmXmlWriter(saveFile, logger)
	writer.appendLog(logger.toString())
	#writer.appendModel(trackmate.getModel())
	#writer.appendSettings(trackmate.getSettings())
	writer.writeToFile();
	print("Results saved to: " + saveFile.toString() + '\n');
	
	# ----------------
	# Display results
	# ----------------
	
	if show_output:
	    model = trackmate.getModel()
	    selectionModel = SelectionModel(model)
	    ds = DisplaySettings()
	    ds = DisplaySettingsIO.readUserDefault()
	    ds.spotDisplayedAsRoi = True
	    displayer = HyperStackDisplayer(model, selectionModel, imp, ds)
	    displayer.render()
	    displayer.refresh()
	
	# capture overlay - RGB file
	image = trackmate.getSettings().imp
	capture = CaptureOverlayAction.capture(image, -1, imp.getNFrames(), logger)
	capture.setTitle("TracksOverlay")
	capture.show()
	
	capture.close()
	capture.flush()
	
	imp.close()
	imp.flush()
	
	IJ.run("Collect Garbage")
	IJ.run("Fresh Start")


print('All done! :)')
