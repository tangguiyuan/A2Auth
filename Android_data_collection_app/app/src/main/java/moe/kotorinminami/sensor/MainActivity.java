package moe.kotorinminami.sensor;


import android.app.Activity;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.ServiceConnection;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Bundle;
import android.os.IBinder;
import android.os.ParcelFileDescriptor;
import android.os.RemoteException;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.RelativeLayout;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;

import java.io.BufferedReader;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.ServerSocket;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Timer;
import java.util.TimerTask;
import java.util.Vector;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

import rikka.shizuku.Shizuku;

public class MainActivity extends Activity implements Runnable, Shizuku.OnRequestPermissionResultListener, ServiceConnection, SeekBar.OnSeekBarChangeListener {
    private TextView showData;
    private TextView rateText;
    private boolean isRunning = false;
    private boolean permissionIsGranted = false;
    private SensorService mService;
    private boolean bound = false;
    private Shizuku.UserServiceArgs mUserServiceArgs;
    private final int port = 11451;
    private static final String TAG = "kotorinminami";
    private Timer timer;

    private int rate = 50;
    private static final int CREATE_FILE = 1;

    private final Vector<String> sensorDataList = new Vector<>();
    private final SensorData sensorData = new SensorData();
    private int touchX = 0;
    private int touchY = 0;
    private int touchPressure = 0;
    // 0: up, 1: down, 2: move
    private int touchAction = 0;
    private int majorAxis = 0;
    private int minorAxis = 0;
    private String nowActivity = "";

    private TimerTask timerTask;
    private IGetEventService iUserService;
    private RelativeLayout lockScreen;
    private Button btnLockScreen;
    private  RelativeLayout normalScreen;

    private Lock lock = new ReentrantLock();
    private final ServiceConnection connection = new ServiceConnection() {
        @Override
        public void onServiceConnected(ComponentName className, IBinder service) {
            SensorService.LocalBinder binder = (SensorService.LocalBinder) service;
            mService = binder.getService();
            bound = true;
        }

        @Override
        public void onServiceDisconnected(ComponentName arg0) {
            bound = false;
        }
    };

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (requestCode == CREATE_FILE && resultCode == Activity.RESULT_OK) {
            Uri uri = data.getData();
            writeDataToFile(uri);
        }
    }

    private void createFile() {
        Intent intent = new Intent(Intent.ACTION_CREATE_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("text/csv");
        intent.putExtra(Intent.EXTRA_TITLE, "sensor_data_" + new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date()) + ".csv");
        startActivityForResult(intent, CREATE_FILE);
    }

    private void writeDataToFile(Uri uri) {
        try {
            lock.lock();
            ParcelFileDescriptor pfd = getContentResolver().openFileDescriptor(uri, "w");
            if (pfd == null) {
                Toast.makeText(this, "Error opening file", Toast.LENGTH_SHORT).show();
                return;
            }
            FileOutputStream fileOutputStream = new FileOutputStream(pfd.getFileDescriptor());

            // 写入CSV头部
            String header = "Date,X,Y,Pressure,Action,MajorAxis,MinorAxis," +
                    "AccelX,AccelY,AccelZ,GravityX,GravityY,GravityZ," +
                    "MagneticX,MagneticY,MagneticZ,activity\n";
            fileOutputStream.write(header.getBytes());

            // 写入数据
            for (String dataLine : sensorDataList) {
                fileOutputStream.write((dataLine).getBytes());
            }

            fileOutputStream.close();
            pfd.close();
            Toast.makeText(this, "File saved successfully", Toast.LENGTH_SHORT).show();
            lock.unlock();
        } catch (IOException e) {
            e.printStackTrace();
            Toast.makeText(this, "Error saving file: " + e.getMessage(),
                    Toast.LENGTH_SHORT).show();
        }
    }

    private boolean requestShizukuPermission() {
        if (Shizuku.pingBinder()) {
            Shizuku.addRequestPermissionResultListener(this);
            Shizuku.requestPermission(0);
            return true;
        }
        else {
            Toast.makeText(this, R.string.need_permission, Toast.LENGTH_SHORT).show();
            return false;
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "onCreate: ");
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);
        showData = findViewById(R.id.touch_data);
        mUserServiceArgs = new Shizuku.UserServiceArgs(new ComponentName(getPackageName(), GetEventService.class.getName()))
                .daemon(false)
                .debuggable(false)
                .processNameSuffix("kotorinminami_sensor")
                .version(1);
        requestShizukuPermission();
        sensorDataList.removeAllElements();
        Button record = findViewById(R.id.btn_record);
        Button export = findViewById(R.id.btn_export);
        SeekBar seekBar = findViewById(R.id.sampling_rate_slider);
        rateText = findViewById(R.id.sampling_rate_value);
        lockScreen = findViewById(R.id.lockscreen);
        btnLockScreen = findViewById(R.id.btn_lockscreen);
        normalScreen = findViewById(R.id.container);

        record.setOnClickListener(
            v -> {
                if (isRunning) {
                    Toast.makeText(this, R.string.end, Toast.LENGTH_SHORT).show();
                    stopService(new Intent(this, SensorService.class));
                    stopTimer();
                    isRunning = false;
                    if (bound) {
                        unbindService(connection);
                        bound = false;
                    }
                    record.setText(R.string.start);
                    return;
                }
                if (!permissionIsGranted) return;
                isRunning = true;
                bindShizukuService();
                startService(new Intent(this, SensorService.class));
                Intent intent = new Intent(this, SensorService.class);
                bindService(intent, connection, Context.BIND_AUTO_CREATE);
                record.setText(R.string.stop);
                Toast.makeText(this, R.string.touch_tips, Toast.LENGTH_SHORT).show();
                sensorDataList.clear();
                startTimer();
            }
        );
        seekBar.setOnSeekBarChangeListener(this);
        export.setOnClickListener(v -> {
            if (sensorDataList.isEmpty()) {
                Toast.makeText(this, "No data to export", Toast.LENGTH_SHORT).show();
                return;
            }
            stopTimer();
            createFile();
            startTimer();
        });
        btnLockScreen.setOnClickListener(v -> {
            if (lockScreen.getVisibility() == View.VISIBLE) {
                lockScreen.setVisibility(View.GONE);
                normalScreen.setVisibility(View.VISIBLE);
                btnLockScreen.setText(R.string.show);
            } else {
                lockScreen.setVisibility(View.VISIBLE);
                normalScreen.setVisibility(View.GONE);
                btnLockScreen.setText(R.string.hide);
            }
        }
        );
    }

    @Override
    protected void onStart() {
        super.onStart();
        // 绑定服务
        Log.i(TAG, "onStart: ");
    }

    @Override
    public void onRequestPermissionResult(int requestCode, int grantResult) {
        permissionIsGranted = grantResult == 0;
    }

    @Override
    protected void onResume() {
        if (Shizuku.pingBinder()) {
            permissionIsGranted = Shizuku.checkSelfPermission() == PackageManager.PERMISSION_GRANTED;
            if (permissionIsGranted) {
                Shizuku.removeRequestPermissionResultListener(this);
            } else {
                Shizuku.addRequestPermissionResultListener(this);
                Shizuku.requestPermission(0);
            }
        }
        else {
            Toast.makeText(this, R.string.need_permission, Toast.LENGTH_SHORT).show();
        }
        super.onResume();
    }

    private void bindShizukuService() {
        if (Shizuku.checkSelfPermission() == PackageManager.PERMISSION_GRANTED | requestShizukuPermission()) {
            new Thread(this).start();
            Shizuku.bindUserService(mUserServiceArgs, this);
        }
    }

    public void showData() {
        showData.setText(
            SensorData.exportToString()
        );
    }

    @Override
    public void run() {
        try (ServerSocket server = new ServerSocket(port)) {
            boolean touchSynState = false;
            while (isRunning) {
                BufferedReader br = new BufferedReader(new InputStreamReader(server.accept().getInputStream()));
                while (isRunning) {
                    String line = br.readLine().strip();
                    // format:/dev/input/event4: EV_ABS       ABS_MT_POSITION_X    00000371
                    if (line.contains("ABS_MT_POSITION_X")) {
                        touchX = Integer.parseInt(line.substring(line.lastIndexOf(" ") + 1), 16);
                        touchAction = touchSynState ? 2 : touchAction;
                    } else if (line.contains("ABS_MT_POSITION_Y")) {
                        touchY = Integer.parseInt(line.substring(line.lastIndexOf(" ") + 1), 16);
                        touchAction = touchSynState ? 2 : touchAction;
                    } else if (line.contains("ABS_MT_PRESSURE")) {
                        touchPressure = Integer.parseInt(line.substring(line.lastIndexOf(" ") + 1), 16);
                    } else if (line.contains("BTN_TOUCH")) {
                        touchAction = line.contains("UP") ? 0 : 1;
                        touchSynState = false;
                    } else if (line.contains("ABS_MT_TOUCH_MAJOR")) {
                        majorAxis = Integer.parseInt(line.substring(line.lastIndexOf(" ") + 1), 16);
                    } else if (line.contains("ABS_MT_TOUCH_MINOR")) {
                        minorAxis = Integer.parseInt(line.substring(line.lastIndexOf(" ") + 1), 16);
                    } else if (line.contains("SYN_REPORT")) {
                        touchSynState = true;
                    } else if (line.contains("From dumpsys")) {
//                        Log.i(TAG, "run: "+line);
                        nowActivity = line.substring(line.lastIndexOf(" ") + 1);
                    }
                    sensorData.update(
                            touchX,
                            touchY,
                            touchPressure,
                            touchAction,
                            majorAxis,
                            minorAxis,
                            mService.getAccelerometerValues(),
                            mService.getGravityValues(),
                            mService.getMagneticValues(),
                            nowActivity
                    );
                    runOnUiThread(this::showData);
//                    Log.i("kotorinminami", line);
                }
            }
        } catch (IOException ignored) {}
//        catch (RemoteException ignored){}
    }

    @Override
    public void onServiceConnected(ComponentName name, IBinder service) {
        if (service != null && service.pingBinder()) {
            iUserService = IGetEventService.Stub.asInterface(service);
            // getevent 进程
            try {
                iUserService.getEvent(port);
            } catch (RemoteException ignored) {
            }
        }
    }

    private void startTimer() {
        timer = new Timer();
        timerTask = new TimerTask() {
            @Override
            public void run() {
                lock.lock();
                if (isRunning && bound && mService != null) {
                    sensorDataList.addElement(SensorData.exportToCSVLine());
//                    Log.i(TAG, "log sensor data");
                }
                lock.unlock();
            }
        };
        timer.schedule(timerTask, 1000l/rate, 1000l / rate);
        Log.i(TAG, "startTimer");
    }

    private void stopTimer() {
        timer.cancel();
        Log.i(TAG, "stopTimer");
    }

    @Override
    public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
        rateText.setText(progress + " Hz");
        rate = progress;
    }

    @Override
    public void onStartTrackingTouch(SeekBar seekBar) {
    }

    @Override
    public void onStopTrackingTouch(SeekBar seekBar) {
    }

    @Override
    protected void onDestroy() {
        isRunning = false;
        if (bound) {
            unbindService(connection);
            bound = false;
        }
        Shizuku.unbindUserService(mUserServiceArgs, this, true);
        stopService(new Intent(this, SensorService.class));
        super.onDestroy();
    }

    @Override
    public void onServiceDisconnected(ComponentName name) {
        isRunning = false;
        iUserService = null;
    }
}