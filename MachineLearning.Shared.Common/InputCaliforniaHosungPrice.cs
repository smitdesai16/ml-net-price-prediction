using Microsoft.ML.Data;

namespace MachineLearning.Shared.Common
{
    public class InputCaliforniaHosungPrice
    {
        //[LoadColumn(0)]
        public float Longitude;

        //[LoadColumn(1)]
        public float Latitude;

        //[LoadColumn(2)]
        public float HousingMedianAge;

        //[LoadColumn(3)]
        public float TotalRooms;

        //[LoadColumn(4)]
        public float TotalBedRooms;

        //[LoadColumn(5)]
        public float Population;

        //[LoadColumn(6)]
        public float HouseHolds;

        //[LoadColumn(7)]
        public float MedianIncome;

        //[LoadColumn(8)]
        public float MedianHouseValue;

        //[LoadColumn(9)]
        public string OceanProximity;
    }
}
