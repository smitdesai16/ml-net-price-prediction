using Microsoft.ML.Data;

namespace MachineLearning.Shared.Common
{
    public class OutputCaliforniaHosungPrice
    {
        [ColumnName("Score")]
        public float MedianHouseValue;
    }
}
