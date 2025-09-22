package moe.kotorinminami.sensor;

import android.os.Build;

import java.text.Format;
import java.text.SimpleDateFormat;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;

public class SensorData {
    private static String date = "";
    private static int touchX;
    private static int touchY;
    private static int touchPressure;
    private static int touchAction;
    private static int majorAxis;
    private static int minorAxis;
    private static float[] accelerometerValues = new float[3];
    private static float[] gravityValues = new float[3];
    private static float[] magneticValues = new float[3];

    private static String activity = "";

    public void update(
            int touchX,
            int touchY,
            int touchPressure,
            int touchAction,
            int majorAxis,
            int minorAxis,
            float[] accelerometerValues,
            float[] gravityValues,
            float[] magneticValues,
            String activity
    ) {
        // 直接使用类变量名,不要用this
        SensorData.date = LocalDateTime.now(ZoneOffset.of("+8")).format(DateTimeFormatter.ofPattern("yyyy-MM-dd-HH:mm:ss.SSSSSS"));
        SensorData.touchX = touchX;
        SensorData.touchY = touchY;
        SensorData.touchPressure = touchPressure;
        SensorData.touchAction = touchAction;
        SensorData.majorAxis = majorAxis;
        SensorData.minorAxis = minorAxis;
        SensorData.accelerometerValues = accelerometerValues;
        SensorData.gravityValues = gravityValues;
        SensorData.magneticValues = magneticValues;
        SensorData.activity = activity;
    }

    public void updateDateTiem() {
        SensorData.date = LocalDateTime.now(ZoneOffset.of("+8")).format(DateTimeFormatter.ofPattern("yyyy-MM-dd-HH:mm:ss.SSSSSS"));;
    }
    public static String exportToString() {
        return  "Date: " + date + "\n" +
                "X: " + touchX + "\n" +
                "Y: " + touchY + "\n" +
                "Pressure: " + touchPressure + "\n" +
                "Action: " + (touchAction == 0 ? "up" : touchAction == 1 ? "down" : "move") + "\n" +
                "MajorAxis: " + majorAxis + "\n" +
                "MinorAxis: " + minorAxis + "\n" +
                "Accelerometer: " + accelerometerValues[0] + ", " + accelerometerValues[1] + ", " + accelerometerValues[2] + "\n" +
                "Gyroscope: " + gravityValues[0] + ", " + gravityValues[1] + ", " + gravityValues[2] + "\n" +
                "Geomagnetic: " + magneticValues[0] + ", " + magneticValues[1] + ", " + magneticValues[2] + "\n" +
                "Activity: " + activity;
    }

    public static String exportToCSVLine() {
        if (accelerometerValues.length != 3 || gravityValues.length != 3 || magneticValues.length != 3) {
            return "";
        }
        return  LocalDateTime.now(ZoneOffset.of("+8")).format(DateTimeFormatter.ofPattern("yyyy-MM-dd-HH:mm:ss.SSSSSS")) + "," +
                touchX + "," +
                touchY + "," +
                touchPressure + "," +
                (touchAction == 0 ? "up" : touchAction == 1 ? "down" : "move") + "," +
                majorAxis + "," +
                minorAxis + "," +
                accelerometerValues[0] + "," + accelerometerValues[1] + "," + accelerometerValues[2] + "," +
                gravityValues[0] + "," + gravityValues[1] + "," + gravityValues[2] + "," +
                magneticValues[0] + "," + magneticValues[1] + "," + magneticValues[2] +
                "," + activity + "\n";
    }
}
