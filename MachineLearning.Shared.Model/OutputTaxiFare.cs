using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace MachineLearning.Shared.Model
{
    public class OutputTaxiFare
    {
        [ColumnName("Score")]
        public float FareAmount;
    }
}
