using Microsoft.ML.Data;

public class HousingPrediction
{
    [ColumnName("Score")]
    public float PredictedHouseValue { get; set; }

    public float[] FeatureContributions { get; set; }
}
