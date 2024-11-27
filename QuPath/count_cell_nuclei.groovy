import qupath.lib.images.ImageData.ImageType

setImageType(ImageType.BRIGHTFIELD_H_E)
setColorDeconvolutionStains('{"Name" : "H&E default", "Stain 1" : "Hematoxylin", "Values 1" : "0.65111 0.70119 0.29049", "Stain 2" : "Eosin", "Values 2" : "0.2159 0.8012 0.5581", "Background" : " 255 255 255"}');

def imageData0 = getCurrentViewer().getImageData();
def stains     = imageData0.getColorDeconvolutionStains();

// Create csv file
def header   = "name, count"
def path     = buildFilePath(PROJECT_BASE_DIR) + '/results.csv'
File csvFile = new File(path)
csvFile.createNewFile()

// Loop through all images in the project
def project = getProject()
def number  = 0

new File(path).withWriter { fw ->
    fw.writeLine(header)
   
    for (entry in project.getImageList()) {
        // Keep track of patch number being analysed
	    print number

        clearAnnotations()
        clearDetections()
        
        // Read image data
        def imageData = entry.readImageData()
        def hierarchy = imageData.getHierarchy()
        def name = entry.getImageName()
        
        // Set image format
        imageData.setImageType(ImageType.BRIGHTFIELD_H_E)
        imageData.setColorDeconvolutionStains(stains)

        // Select detection zone - square (512x512 pixels) 
        def plane     = ImagePlane.getPlane(0, 0)
        def roi       = ROIs.createRectangleROI(0, 0, 512,512, plane)
        def object    = PathObjects.createAnnotationObject(roi)
        
        // Add to hierarchy
        hierarchy.addPathObjects([object])
        
        // Select rectangle annotation
        def annotation = hierarchy.getAnnotationObjects()[0]
        hierarchy.getSelectionModel().clearSelection()
        hierarchy.getSelectionModel().setSelectedObject(annotation)   
    
        // Run cell detection
        runPlugin('qupath.imagej.detect.cells.PositiveCellDetection', imageData, '{"detectionImageBrightfield":"Hematoxylin OD","backgroundByReconstruction":true,"backgroundRadius":15.0,"medianRadius":0.5,"sigma":2.0,"minArea":15.0,"maxArea":150.0,"threshold":0.02,"maxBackground":2.0,"watershedPostProcess":true,"cellExpansion":1.0,"includeNuclei":true,"smoothBoundaries":true,"makeMeasurements":true,"thresholdCompartment":"Nucleus: Eosin OD mean","thresholdPositive1":0.25,"thresholdPositive2":0.4,"thresholdPositive3":0.6000000000000001,"singleThreshold":true}')
        
        // Get 'Positive' detections
        positive = hierarchy.getDetectionObjects().findAll{it.getPathClass().toString().contains("Positive")}
        count    = positive.size()
        
        // Save cell count
        String line = name + "," + count
        fw.writeLine(line)
        
	number = number + 1
    }
}

print 'DONE!'