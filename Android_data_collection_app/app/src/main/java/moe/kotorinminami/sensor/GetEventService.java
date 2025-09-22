package moe.kotorinminami.sensor;

import android.util.Log;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.InetAddress;
import java.net.Socket;

public class GetEventService extends IGetEventService.Stub implements Runnable {
    private final ProcessBuilder builder = new ProcessBuilder();
    private boolean isServiceRunning = true;
    private int port;
    private int errCount;

    @Override
    public void destroy() {
        isServiceRunning = false;
    }

    @Override
    public void getEvent(int port) {
        this.port = port;
        new Thread(this).start();
    }

    @Override
    public String getActivity() {
        String activity = "";
        ProcessBuilder testBuilder = new ProcessBuilder();
        try {
            String[] cmdline = {"dumpsys", "window"};
            Process process = testBuilder.command(cmdline).start();
            BufferedReader bs = new BufferedReader(new InputStreamReader(process.getInputStream()));
            String line;
            while ((line = bs.readLine()) != null) {
                if (line.contains("mCurrentFocus")) {
                    line = line.trim().replaceAll("\\}+$", "");
                    Log.i("kotorishizuku", "getActivity: "+line);
                    activity = line.substring(line.lastIndexOf(" ") + 1);
                    break;
                }
            }
        } catch (IOException ignored) {
        }
        return activity;
    }

    /**
     * @noinspection BusyWait
     */
    @Override
    public void run() {
        while (isServiceRunning) {
            try (Socket socket = new Socket(InetAddress.getLocalHost(), port)) {
                String[] cmdline = {"getevent", "-l"};
                InputStream in = builder.command(cmdline).start().getInputStream();
                OutputStream out = socket.getOutputStream();
                int ch;
                while (isServiceRunning) {
                    if (in.available() > 0) {
                        ch = in.read();
                        if (ch == -1) {
                            Thread.sleep(1);
                        } else {
                            out.write(ch);
                        }
                    }
                    else {
                        String activity = "";
                        ProcessBuilder testBuilder = new ProcessBuilder();
                        try {
                            String[] cmd = {"dumpsys", "window"};
                            Process process = testBuilder.command(cmd).start();
                            BufferedReader bs = new BufferedReader(new InputStreamReader(process.getInputStream()));
                            String line;
                            while ((line = bs.readLine()) != null) {
                                if (line.contains("mCurrentFocus")) {
                                    line = line.trim().replaceAll("\\}+$", "");
//                                    Log.i("kotorishizuku", "getActivity: "+line);
                                    activity = "From dumpsys " + line.substring(line.lastIndexOf(" ") + 1) + "\n";
                                    break;
                                }
                            }
                            activity.codePoints().forEach(
                                    c -> {
                                        try {
                                            out.write(c);
                                        } catch (IOException e) {
                                        }
                                    }
                            );
                        } catch (IOException ignored) {
                        }
                    }
                }
            } catch (IOException | InterruptedException ignored) {
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException ignored1) {
                }
                errCount++;
                if (errCount > 15) {
                    isServiceRunning = false;
                }
            }
        }
    }
}