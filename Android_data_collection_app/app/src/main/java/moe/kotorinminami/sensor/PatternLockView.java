package moe.kotorinminami.sensor;
import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Path;
import android.util.AttributeSet;
import android.util.Log;
import android.util.Pair;
import android.view.MotionEvent;
import android.view.View;

import java.util.ArrayList;
import java.util.List;

public class PatternLockView extends View {
    private List<Pair<Integer, Integer>> selectedPoints = new ArrayList<>();
    private Paint paint = new Paint();
    private Paint connectorPaint = new Paint(); 
    private Path path = new Path();
    private float[][][] positions; 
    private int nodeRadius = 100;
    private float currentX, currentY; 
    private String TAG = "lock";
    private boolean UP = false;

    public PatternLockView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    private void init() {
        paint.setColor(0xff000000);
        paint.setStrokeWidth(20f);
        paint.setStyle(Paint.Style.STROKE);
        paint.setAntiAlias(true);
        paint.setStrokeCap(Paint.Cap.ROUND);

        connectorPaint.setColor(0x88000000); 
        connectorPaint.setStrokeWidth(15f);
        connectorPaint.setStyle(Paint.Style.STROKE);
        connectorPaint.setAntiAlias(true);

        positions = new float[3][3][2];

        Log.i(TAG, "init: lock view");
    }

    @Override
    protected void onMeasure(int widthMeasureSpec, int heightMeasureSpec) {
        int desiredWidth = (int) (300 * getResources().getDisplayMetrics().density);
        int desiredHeight = (int) (300 * getResources().getDisplayMetrics().density);

        int widthMode = MeasureSpec.getMode(widthMeasureSpec);
        int widthSize = MeasureSpec.getSize(widthMeasureSpec);
        int heightMode = MeasureSpec.getMode(heightMeasureSpec);
        int heightSize = MeasureSpec.getSize(heightMeasureSpec);

        int width;
        if (widthMode == MeasureSpec.EXACTLY) {
            width = widthSize;
        } else if (widthMode == MeasureSpec.AT_MOST) {
            width = Math.min(desiredWidth, widthSize);
        } else {
            width = desiredWidth;
        }

        int height;
        if (heightMode == MeasureSpec.EXACTLY) {
            height = heightSize;
        } else if (heightMode == MeasureSpec.AT_MOST) {
            height = Math.min(desiredHeight, heightSize);
        } else {
            height = desiredHeight;
        }

        setMeasuredDimension(width, height);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                positions[i][j][0] = width / 4f * (2 + (i - 1));
                positions[i][j][1] = height / 4f * (2 + (j - 1));
            }
        }
    }


    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        // 绘制解锁点圆圈

        for (int row = -1; row <= 1; row++) {
            for (int col = -1; col <= 1; col++) {
                canvas.drawPoint(positions[row + 1][col + 1][0], positions[row + 1][col + 1][1], paint);
            }
        }

        if (!selectedPoints.isEmpty()){
            for (int i = 0; i < selectedPoints.size(); i++) {
                Pair<Integer, Integer> start = selectedPoints.get(i);
                if (i < selectedPoints.size() - 1){
                    Pair<Integer, Integer> end = selectedPoints.get(i+1);
                    canvas.drawLine(positions[start.first][start.first][0], positions[start.first][start.second][1], positions[end.first][end.second][0], positions[end.first][end.second][1], connectorPaint);
                }
                else if (!UP){
                    canvas.drawLine(positions[start.first][start.second][0], positions[start.first][start.second][1], currentX, currentY, connectorPaint);
                }
            }
        }
    }


    @Override
    public boolean onTouchEvent(MotionEvent event) {
        currentX = event.getX();
        currentY = event.getY();

        switch (event.getAction()) {
            case MotionEvent.ACTION_DOWN:
                selectedPoints.clear();
                path.reset();
                UP = false;
                Pair<Integer, Integer> index = getIndexAtPosition(currentX, currentY);
                if (index.first != -1){
                    selectedPoints.add(index);
                    invalidate();
                    return true;
                }
            case MotionEvent.ACTION_MOVE:
                Pair<Integer, Integer> moveIndex = getIndexAtPosition(currentX, currentY);
                if (moveIndex.first != -1 && !selectedPoints.contains(moveIndex)){
                    selectedPoints.add(moveIndex);
                }
                invalidate();
                return true;
            case MotionEvent.ACTION_UP:
                UP = true;
                invalidate();
                return true;

        }
        return super.onTouchEvent(event);
    }


    private Pair<Integer, Integer> getIndexAtPosition(float x, float y) {
        // 计算点击位置对应的解锁点
        for (int i=0; i<3; i++){
            for (int j=0; j<3; j++){
                float nodeX = positions[i][j][0];
                float nodeY = positions[i][j][1];
                double distance = Math.sqrt(Math.pow(x - nodeX, 2) + Math.pow(y - nodeY, 2));
                if (distance < nodeRadius){
                    return new Pair<>(i, j);
                }
            }
        }
        return new Pair<>(-1, -1);
    }

    private String getPatternString(List<Integer> pattern) {
        StringBuilder stringBuilder = new StringBuilder();
        for (Integer node : pattern) {
            stringBuilder.append(node);
        }
        return stringBuilder.toString();
    }

    public interface OnPatternListener {
        void onPatternComplete(String pattern);
    }
    private OnPatternListener listener;
    public void setOnPatternListener(OnPatternListener listener){
        this.listener = listener;
    }

    public void onPatternComplete(String pattern) {
        if(listener != null) {
            listener.onPatternComplete(pattern);
        }
        invalidate(); 
    }

}