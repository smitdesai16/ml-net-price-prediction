using Microsoft.ML.Data;

namespace MachineLearning.Shared.Model
{
    public class OutputCaliforniaHosungPrice
    {
        [ColumnName("Score")]
        public float MedianHouseValue;
    }
}
