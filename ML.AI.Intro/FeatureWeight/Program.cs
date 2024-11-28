using Microsoft.ML;

const string housingData = "housing.csv";

var context = new MLContext();
var data = context.Data.LoadFromTextFile<HousingData>(housingData, hasHeader: true, separatorChar: ',');

var featureColumns = data.Schema
    .Select(col => col.Name)
    .Where(colName => colName != "Label" && colName != "OceanProximity")
    .ToArray();

var pipeline = context.Transforms.Concatenate("Features", featureColumns)
    .Append(context.Regression.Trainers.LbfgsPoissonRegression());

var model = pipeline.Fit(data);
var transformedData = model.Transform(data);

// Get weights of model
var linearModel = model.LastTransformer.Model;

var weights = linearModel.Weights;

Console.WriteLine("Weights:");
// Order features by importance
var featureWeights = weights
                    .Select((item, index) => new { index, item })
                    .OrderByDescending(theWeight => Math.Abs(theWeight.item));

foreach (var featureWeight in featureWeights)
{
    Console.WriteLine($"Feature {featureColumns[featureWeight.index]} has weight {Math.Abs(featureWeight.item)}");
}
Console.WriteLine(Environment.NewLine);
