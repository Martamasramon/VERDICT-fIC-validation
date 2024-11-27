Imports MediaCy.Addins.DataCollector.Gadgets
Imports MediaCy.Addins.Scripting
Imports MediaCy.Addins.Measurements.Gadgets
Imports MediaCy.Addins.LearningSegmentation
Imports MediaCy.Addins.Measurements
Imports System.IO

Public Module Module1
    Public Function Segment_tissue() As SimpleScript
        Segment_tissue = New SimpleScript
        Dim var1 = ThisApplication.Path(mcPathType.mcptProgram) & "Welcome\Welcome.htm"
        Dim doc1
        Dim docList1 = New List(1), doc2, image1, doc3, window1, window2

        ' Specify folders
        Dim folderInPath  As String = "F:\Histology\Patches\"
        Dim folderOutPath As String = "F:\Histology\Masks\"

        Dim imageFiles() As String = Directory.GetFiles(folderInPath, "*.png")

        ' Loop through each image file
        For Each imagePath As String In imageFiles

            ' Open the current image file
            With Application.DocumentCommands.Open(Segment_tissue)
                .Filenames = New String() {imagePath}
                .Run(docList1)
            End With

            ' Activate the document representing the current image
            With Application.DocumentCommands.Activate(Segment_tissue)
                .Run(docList1(0), doc2)
            End With

            ' Open the segmentation settings file
            With Measure.SmartSegmentation.RecipeCommands.Open(Segment_tissue)
                .FileName = "F:\Histology\segmentation_settings.isg"
                .FilterIndex = 1
                .Settings = Nothing
                .FilterInput = True
                .Run(doc2)
            End With

            ' Create mask for the current image
            With Measure.SmartSegmentationCommands.CreateMask(Segment_tissue)
                .Binary = False
                .FilterInput = True
                .Run(doc2, image1)
            End With

            ' Activate the document representing the mask
            With Application.DocumentCommands.Activate(Segment_tissue)
                .Run(image1, doc3)
            End With

            ' Save the mask as a PNG file
            Dim maskFilePath As String = Path.Combine(folderOutPath, Path.GetFileNameWithoutExtension(imagePath) & "_mask.png")
            With Application.DocumentCommands.SaveAs(Segment_tissue)
                .Filename = maskFilePath
                .Run(image1)
            End With

            ' Close the documents
            With Application.WindowCommands.Close(Segment_tissue)
                .Run(window1)
            End With

            With Application.WindowCommands.Close(Segment_tissue)
                .Run(window2)
            End With
        Next

    End Function

End Module
