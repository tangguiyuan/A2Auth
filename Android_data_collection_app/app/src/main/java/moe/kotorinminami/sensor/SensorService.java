package moe.kotorinminami.sensor;


import java.text.SimpleDateFormat;

import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.IBinder;
import android.os.PowerManager;
import android.os.PowerManager.WakeLock;
import android.util.Log;

public class SensorService extends Service {
    private static final String TAG = "DeviceSensorService";
    Sensor sensorAcc, sensoGyros, sensoGeo;
    float[] accelerometerValues;
    float[] gravityValues;
    float[] magneticValues;
    SensorManager sm;
    WakeLock m_wklk;

    public class LocalBinder extends android.os.Binder {
        public SensorService getService() {
            return SensorService.this;
        }
    }

    private final IBinder mBinder = new LocalBinder();

    @Override
    public void onCreate() {
        super.onCreate();
        if (sm == null) {
            sm = (SensorManager) getApplicationContext().getSystemService(
                    Context.SENSOR_SERVICE);
        }

        Sensor sensorAcc = sm.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        Sensor sensoGyros = sm.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        Sensor sensoGeo = sm.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD);

        sm.registerListener(mySensorListener, sensorAcc,
                SensorManager.SENSOR_DELAY_FASTEST); 
        sm.registerListener(mySensorListener, sensoGyros,
                SensorManager.SENSOR_DELAY_FASTEST);
        sm.registerListener(mySensorListener, sensoGeo,
                SensorManager.SENSOR_DELAY_FASTEST);

        float[] accelerometerValues = new float[3];
        float[] gravityValues = new float[3];
        float[] magneticValues = new float[3];

        PowerManager pm = (PowerManager) getSystemService(Context.POWER_SERVICE);
        m_wklk = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, SensorService.class.getName());
        m_wklk.acquire();

    }

    @Override
    public IBinder onBind(Intent intent) {
        return mBinder;
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        Log.i(TAG, "onStartCommand");
        return Service.START_STICKY;
    }

    public void onDestroy() {
        if (sm != null) {
            sm.unregisterListener(mySensorListener);
            mySensorListener = null;
        }
        if (m_wklk != null) {
            m_wklk.release();
            m_wklk = null;
        }
    };

    private SensorEventListener mySensorListener = new SensorEventListener() {

        public void onSensorChanged(SensorEvent sensorEvent) {
            synchronized (this) {
                int type = sensorEvent.sensor.getType();

                SimpleDateFormat sDateFormat = new SimpleDateFormat("yyyy-MM-dd hh:mm:ss:SSS ");
                String date = sDateFormat.format(new java.util.Date());

                switch (type) {
                    case Sensor.TYPE_ACCELEROMETER:
                        accelerometerValues = sensorEvent.values;
//                        Log.i("sensor", "Accelerometer:"+ date + ", "
//                                + accelerometerValues[0] + ","
//                                + accelerometerValues[1] + ","
//                                + accelerometerValues[2] + ";");

                        break;
                    case Sensor.TYPE_GYROSCOPE:
                        gravityValues = sensorEvent.values;
//                        Log.i("sensor", "Gyproscope:"+ date + ", "
//                                + gravityValues[0] + ","
//                                + gravityValues[1] + ","
//                                + gravityValues[2] + ";");
                        break;
                    case Sensor.TYPE_MAGNETIC_FIELD:
                        magneticValues = sensorEvent.values;
//                        Log.i("sensor", "Geomagnetic:"+ date + ","
//                                + magneticValues[0] + ","
//                                + magneticValues[1] + ","
//                                + magneticValues[2]+ ";");
                        break;
                    default:
                        break;
                }
            }
        }

        public void onAccuracyChanged(Sensor sensor, int accuracy) {
            Log.i("sensor", "onAccuracyChanged-----sensor"+ sensor + ",acc:" + accuracy);

        }
    };

    public float[] getAccelerometerValues() {
        return accelerometerValues;
    }

    public float[] getGravityValues() {
        return gravityValues;
    }

    public float[] getMagneticValues() {
        return magneticValues;
    }
}
