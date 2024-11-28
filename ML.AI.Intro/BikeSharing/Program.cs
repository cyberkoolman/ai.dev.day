using Microsoft.ML;
using System.Text.Json;

const string TrainingDataLocation = "Data/hour_train.csv";
const string TestDataLocation = "Data/hour_test.csv";
const string ModelFilename = "_model.zip";

// Create MLContext to be shared across the model creation workflow objects 
// Set a random seed for repeatable/deterministic results across multiple trainings.
var mlContext = new MLContext(seed: 0);

// STEP 1. Load Data
var trainingDataView = mlContext.Data.LoadFromTextFile<DemandObservation>(path: TrainingDataLocation, hasHeader: true, separatorChar: ',');
var testDataView = mlContext.Data.LoadFromTextFile<DemandObservation>(path: TestDataLocation, hasHeader: true, separatorChar: ',');

// STEP 2. Featurize Data
// Concatenate all the numeric columns into a single features column
// Question: 
//    1. Why temperature, humidity, windspeed values are normalized?
//    2. Why instant, dteday are not part of features?
//    3. Why casual, registered are not part of features?
var dataProcessPipeline = mlContext.Transforms.Concatenate("Features",
                nameof(DemandObservation.Season), 
                nameof(DemandObservation.Year), 
                nameof(DemandObservation.Month),
                nameof(DemandObservation.Hour), 
                nameof(DemandObservation.Holiday), 
                nameof(DemandObservation.Weekday),
                nameof(DemandObservation.WorkingDay), 
                nameof(DemandObservation.Weather), 
                nameof(DemandObservation.Temperature),
                nameof(DemandObservation.NormalizedTemperature), 
                nameof(DemandObservation.Humidity), 
                nameof(DemandObservation.Windspeed))
                .AppendCacheCheckpoint(mlContext);

// Regression trainers/algorithms to use
// var regressionLearners = new (string nameOfAlgorithm, IEstimator<ITransformer> estimator)[]
(string name, IEstimator<ITransformer> value)[] regressionLearners =
{
    ("FastTree", mlContext.Regression.Trainers.FastTree()),
    ("Poisson", mlContext.Regression.Trainers.LbfgsPoissonRegression()),
    ("SDCA", mlContext.Regression.Trainers.Sdca()),
    ("FastTreeTweedie", mlContext.Regression.Trainers.FastTreeTweedie()),
    // Other possible learners that could be included
    //...FastForestRegressor...
    //...GeneralizedAdditiveModelRegressor...
    //...OnlineGradientDescent... (Might need to normalize the features first)
};

// STEP 3.  Training, Evaluation and model file persistence
// Per each regression trainer: Train, Evaluate, and Save a different model
foreach (var trainer in regressionLearners)
{
    Console.WriteLine("=============== Training the current model ===============");
    var trainingPipeline = dataProcessPipeline.Append(trainer.value);
    var trainedModel = trainingPipeline.Fit(trainingDataView);

    Console.WriteLine("===== Evaluating Model's accuracy with Test data =====");
    IDataView predictions = trainedModel.Transform(testDataView);
    var metrics = mlContext.Regression.Evaluate(data: predictions, labelColumnName: "Label", scoreColumnName: "Score");

    Console.WriteLine($"*************************************************");
    Console.WriteLine($"*       Metrics for {trainer.name} regression model      ");
    Console.WriteLine($"*------------------------------------------------");
    Console.WriteLine($"*       LossFn:        {metrics.LossFunction:0.##}");
    Console.WriteLine($"*       R2 Score:      {metrics.RSquared:0.##}");
    Console.WriteLine($"*       Absolute loss: {metrics.MeanAbsoluteError:#.##}");
    Console.WriteLine($"*       Squared loss:  {metrics.MeanSquaredError:#.##}");
    Console.WriteLine($"*       RMS loss:      {metrics.RootMeanSquaredError:#.##}");
    Console.WriteLine($"*************************************************");

    //Save the model file that can be used by any application
    string modelPath = $"{trainer.name}{ModelFilename}";
    mlContext.Model.Save(trainedModel, trainingDataView.Schema, modelPath);
    Console.WriteLine("The model is saved to {0}", modelPath);
}

// STEP 4. Test Predictions with the created models
string jsonString = JsonSerializer.Serialize(DemandObservationSample.SingleDemandSampleData, new JsonSerializerOptions
{
    WriteIndented = true // Pretty print the JSON
});
Console.WriteLine("===== Prediction for the date & condition below =====");
Console.WriteLine(jsonString);
Console.WriteLine($"*************************************************");

foreach (var learner in regressionLearners)
{
    //Load current model
    string modelPath = $"{learner.name}{ModelFilename}";
    ITransformer trainedModel = mlContext.Model.Load(modelPath, out var modelInputSchema);
    var predEngine = mlContext.Model.CreatePredictionEngine<DemandObservation, DemandPrediction>(trainedModel);

    var sample = DemandObservationSample.SingleDemandSampleData;
    var prediction = predEngine.Predict(sample);
    Console.WriteLine($"{learner.name}:   {prediction.PredictedCount}");
}

