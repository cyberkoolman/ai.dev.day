using System.IO;
using Microsoft.ML;

bool isTrainingMode = false;
if (args.Length > 0 && args[0].Equals("train", StringComparison.OrdinalIgnoreCase))
{
    isTrainingMode = true;
}

var mlContext = new MLContext();

string TrainDataPath = $"Data/HeartTraining.csv";
string TestDataPath = $"Data/HeartTest.csv";
string ModelPath = $"./HeartClassification.zip";

if (isTrainingMode)
{
    // STEP 1: Common data loading configuration
    var trainingDataView = mlContext.Data.LoadFromTextFile<HeartData>(TrainDataPath, hasHeader: true, separatorChar: ';');
    var testDataView = mlContext.Data.LoadFromTextFile<HeartData>(TestDataPath, hasHeader: true, separatorChar: ';');

    // STEP 2: Concatenate the features and set the training algorithm
    var pipeline = mlContext.Transforms.Concatenate("Features", "Age", "Sex", "Cp", "TrestBps", "Chol", "Fbs", "RestEcg", "Thalac", "Exang", "OldPeak", "Slope", "Ca", "Thal")
        .Append(mlContext.BinaryClassification.Trainers.FastTree(labelColumnName: "Label", featureColumnName: "Features"));

    Console.WriteLine("=============== Training the model ===============");
    ITransformer trainedModel = pipeline.Fit(trainingDataView);
    Console.WriteLine("");
    Console.WriteLine("=============== Finish the train model. Push Enter ===============");
    Console.WriteLine("");

    Console.WriteLine("===== Evaluating Model's accuracy with Test data =====");
    var predictions = trainedModel.Transform(testDataView);

    var metrics = mlContext.BinaryClassification.Evaluate(data: predictions, labelColumnName: "Label", scoreColumnName: "Score");
    Console.WriteLine("");
    Console.WriteLine($"************************************************************");
    Console.WriteLine($"*       Metrics for {trainedModel.ToString()} binary classification model      ");
    Console.WriteLine($"*-----------------------------------------------------------");
    Console.WriteLine($"*       Accuracy: {metrics.Accuracy:P2}");
    Console.WriteLine($"*       Area Under Roc Curve:      {metrics.AreaUnderRocCurve:P2}");
    Console.WriteLine($"*       Area Under PrecisionRecall Curve:  {metrics.AreaUnderPrecisionRecallCurve:P2}");
    Console.WriteLine($"*       F1Score:  {metrics.F1Score:P2}");
    Console.WriteLine($"*       LogLoss:  {metrics.LogLoss:#.##}");
    Console.WriteLine($"*       LogLossReduction:  {metrics.LogLossReduction:#.##}");
    Console.WriteLine($"*       PositivePrecision:  {metrics.PositivePrecision:#.##}");
    Console.WriteLine($"*       PositiveRecall:  {metrics.PositiveRecall:#.##}");
    Console.WriteLine($"*       NegativePrecision:  {metrics.NegativePrecision:#.##}");
    Console.WriteLine($"*       NegativeRecall:  {metrics.NegativeRecall:P2}");
    Console.WriteLine($"************************************************************");
    Console.WriteLine("");

    Console.WriteLine("=============== Saving the model to a file ===============");
    mlContext.Model.Save(trainedModel, trainingDataView.Schema, ModelPath);
    Console.WriteLine("=============== Model Saved ============= ");
} else {
    if (File.Exists(ModelPath))
    {
        ITransformer trainedModel = mlContext.Model.Load(ModelPath, out var modelInputSchema);

        // Create prediction engine related to the loaded trained model
        var predictionEngine = mlContext.Model.CreatePredictionEngine<HeartData, HeartPrediction>(trainedModel);                   

        foreach (var heartData in HeartSampleData.heartDataList)
        {
            var prediction = predictionEngine.Predict(heartData);

            Console.WriteLine($"=============== Single Prediction  ===============");
            Console.WriteLine($"Age: {heartData.Age} ");
            Console.WriteLine($"Sex: {heartData.Sex} ");
            Console.WriteLine($"Cp: {heartData.Cp} ");
            Console.WriteLine($"TrestBps: {heartData.TrestBps} ");
            Console.WriteLine($"Chol: {heartData.Chol} ");
            Console.WriteLine($"Fbs: {heartData.Fbs} ");
            Console.WriteLine($"RestEcg: {heartData.RestEcg} ");
            Console.WriteLine($"Thalac: {heartData.Thalac} ");
            Console.WriteLine($"Exang: {heartData.Exang} ");
            Console.WriteLine($"OldPeak: {heartData.OldPeak} ");
            Console.WriteLine($"Slope: {heartData.Slope} ");
            Console.WriteLine($"Ca: {heartData.Ca} ");
            Console.WriteLine($"Thal: {heartData.Thal} ");
            Console.WriteLine($"Prediction Value: {prediction.Prediction} ");
            Console.WriteLine($"Prediction: {(prediction.Prediction ? "A disease could be present" : "Not present disease" )} ");
            Console.WriteLine($"Probability: {prediction.Probability} ");
            Console.WriteLine($"==================================================");
        }
    } else {
        Console.WriteLine("You need to train the model first, run with 'train' first");        
    }
}
