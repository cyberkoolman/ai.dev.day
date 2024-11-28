using Microsoft.ML;

string _assetsPath = Path.Combine(Environment.CurrentDirectory, "assets");
string _trainImagesFolder = Path.Combine(_assetsPath, "inputs-train", "data");
string _trainTagsTsv = Path.Combine(_assetsPath, "inputs-train", "data", "tags.tsv");
string _tfModelFile = Path.Combine(_assetsPath, "inputs-train", "inception", "tensorflow_inception_graph.pb");
string _mnModelFile = Path.Combine(_assetsPath, "outputs", "imageClassifier.zip");

string LabelTokey = nameof(LabelTokey);
string PredictedLabelValue = nameof(PredictedLabelValue);

MLContext mlContext = new MLContext(seed: 1);

if (args.Length > 0) {
  var model = mlContext.Model.Load(_mnModelFile, out var _);
  if (args[0].ToLower() == "folder") {
    // Prediction in all images in the folder
    string _predictImagesFolder = Path.Combine(_assetsPath, "inputs-predict", "data");
    string _predictImageListTsv = Path.Combine(_assetsPath, "inputs-predict", "data", "image_list.tsv");

    var imageData = ReadFromTsv(_predictImageListTsv, _predictImagesFolder);
    var imageDataView = mlContext.Data.LoadFromEnumerable<ImageData>(imageData);

    var preds = model.Transform(imageDataView);
    var imagePredData = mlContext.Data.CreateEnumerable<ImagePrediction>(preds, false, true);

    Console.WriteLine("=============== Making classifications ===============");
    DisplayResults(imagePredData);    
  } else {
    // Prediction for one particular image given from the prompt
    string _predictSingleImage = args[0];
    var testImageData = new ImageData()
    {
        ImagePath = Path.Combine(Environment.CurrentDirectory, _predictSingleImage)
    };
    var predictor = mlContext.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);
    var prediction = predictor.Predict(testImageData);

    Console.WriteLine("=============== Making single image classification ===============");
    Console.WriteLine($"Image: {Path.GetFileName(testImageData.ImagePath)}"
            + $"    predicted as: {prediction.PredictedLabelValue}"
            + $"    with score: {prediction.Score.Max()} ");

  };
  return;
};

var trainData = mlContext.Data.LoadFromTextFile<ImageData>(path: _trainTagsTsv, hasHeader: false);

// First, the image transforms transform the images into the model's expected format.
// The ScoreTensorFlowModel transform scores the TensorFlow model and allows communication 
var imgProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: LabelTokey, inputColumnName: "Label")
    .Append(mlContext.Transforms.LoadImages(outputColumnName: "input", 
            imageFolder: _trainImagesFolder, 
            inputColumnName: nameof(ImageData.ImagePath)))
    .Append(mlContext.Transforms.ResizeImages(outputColumnName: "input", 
            imageWidth: InceptionSettings.ImageWidth, 
            imageHeight: InceptionSettings.ImageHeight, 
            inputColumnName: "input"))
    .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "input", 
            interleavePixelColors: InceptionSettings.ChannelsLast, 
            offsetImage: InceptionSettings.Mean))
    .Append(mlContext.Model.LoadTensorFlowModel(_tfModelFile)
        .ScoreTensorFlowModel(outputColumnNames: new[] { "softmax2_pre_activation" }, 
                              inputColumnNames: new[] { "input" }, 
                              addBatchDimensionInput: true))
    .Append(mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(
            labelColumnName: LabelTokey, 
            featureColumnName: "softmax2_pre_activation"))
    .Append(mlContext.Transforms.Conversion.MapKeyToValue(PredictedLabelValue, "PredictedLabel"))
    .AppendCacheCheckpoint(mlContext);

Console.WriteLine("=============== Training classification model ===============");
ITransformer trainedModel = imgProcessPipeline.Fit(trainData);
mlContext.Model.Save(trainedModel, trainData.Schema, _mnModelFile);
Console.WriteLine($"Saved this model to {_mnModelFile}");

var predictions = trainedModel.Transform(trainData);
var imagePredictionData = mlContext.Data.CreateEnumerable<ImagePrediction>(predictions, false, true);
DisplayResults(imagePredictionData);

Console.WriteLine("=============== Classification metrics ===============");
var multiclassContext = mlContext.MulticlassClassification;
var metrics = multiclassContext.Evaluate(predictions, labelColumnName: LabelTokey, predictedLabelColumnName: "PredictedLabel");

Console.WriteLine($"LogLoss is: {metrics.LogLoss}");
Console.WriteLine($"PerClassLogLoss is: {String.Join(" , ", metrics.PerClassLogLoss.Select(c => c.ToString()))}");

IEnumerable<ImageData> ReadFromTsv(string file, string folder)
{
    //Need to parse through the tags.tsv file to combine the file path to the 
    // image name for the ImagePath property so that the image file can be found.
    return File.ReadAllLines(file)
      .Select(line => line.Split('\t'))
      .Select(line => new ImageData()
      {
          ImagePath = Path.Combine(folder, line[0])
      });
}

void DisplayResults(IEnumerable<ImagePrediction> imagePredictionData)
{
    foreach (ImagePrediction prediction in imagePredictionData)
    {
        Console.WriteLine($"Image: {Path.GetFileName(prediction.ImagePath)}"
           + $"    predicted as: {prediction.PredictedLabelValue}"
           + $"    with score: {prediction.Score.Max()} ");
    }
}

struct InceptionSettings
{
    public const int ImageHeight = 224;
    public const int ImageWidth = 224;
    public const float Mean = 117;
    public const float Scale = 1;
    public const bool ChannelsLast = true;
}
