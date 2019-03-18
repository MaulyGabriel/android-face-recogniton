package org.bytedeco.javacv.android.recognize.example;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Context;
import android.content.pm.PackageManager;
import android.os.AsyncTask;
import android.os.Build;
import android.os.Bundle;
import android.os.SystemClock;
import android.support.annotation.Nullable;
import android.support.v4.app.ActivityCompat;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;

import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.opencv_face;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.Point;
import org.bytedeco.javacpp.opencv_core.RectVector;
import org.bytedeco.javacpp.opencv_core.Size;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;

import static org.bytedeco.javacpp.opencv_core.FONT_HERSHEY_PLAIN;
import static org.bytedeco.javacpp.opencv_core.LINE_8;
import static org.bytedeco.javacpp.opencv_core.Mat;
import static org.bytedeco.javacpp.opencv_imgproc.CV_BGR2GRAY;
import static org.bytedeco.javacpp.opencv_imgproc.cvtColor;
import static org.bytedeco.javacpp.opencv_imgproc.putText;
import static org.bytedeco.javacpp.opencv_imgproc.rectangle;
import static org.bytedeco.javacpp.opencv_imgproc.resize;
import static org.bytedeco.javacpp.opencv_objdetect.CascadeClassifier;
import static org.bytedeco.javacv.android.recognize.example.TrainHelper.TOLERANCE_LEVEL;


public class FaceOperatorRecognition extends Activity implements CvCameraPreview.CvCameraViewListener {

    public static final String TAG = "RECOGNITION::ACTIVITY";
    private CascadeClassifier faceDetector;
    private int absoluteFaceSize = 0;
    private CvCameraPreview cameraView;
    public static final String DATASET = "/dataset/operators.txt";
    boolean takePhoto;
    opencv_face.FaceRecognizer faceRecognizer = opencv_face.EigenFaceRecognizer.create();
    opencv_face.FaceRecognizer lbphRecognizer = opencv_face.LBPHFaceRecognizer.create(2, 8, 8, 8, 200);
    boolean trained;
    private String[] operators;
    private int[] myCodes = {3520, 4528, 1025};
    private String[] myOperators = {"Gabriel", "Rene", "Lester"};

    private boolean hasPermissions(Context context, String... permissions) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M && context != null && permissions != null) {
            for (String permission : permissions) {
                if (ActivityCompat.checkSelfPermission(context, permission) != PackageManager.PERMISSION_GRANTED) {
                    return false;
                }
            }
        }
        return true;
    }

    Button btn_capture;
    Button btn_train;
    Button btn_reset;

    @SuppressLint("StaticFieldLeak")
    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_opencv);

        if (Build.VERSION.SDK_INT >= 23) {
            String[] PERMISSIONS = {android.Manifest.permission.READ_EXTERNAL_STORAGE, android.Manifest.permission.WRITE_EXTERNAL_STORAGE};
            if (!hasPermissions(this, PERMISSIONS)) {
                ActivityCompat.requestPermissions(this, PERMISSIONS, 1);
            }
        }

        cameraView = (CvCameraPreview) findViewById(R.id.camera_view);
        cameraView.setCvCameraViewListener(this);

        btn_capture = (Button) findViewById(R.id.btPhoto);
        btn_train = (Button) findViewById(R.id.btTrain);
        btn_reset = (Button) findViewById(R.id.btReset);

        new AsyncTask<Void, Void, Void>() {
            @Override
            protected Void doInBackground(Void... voids) {
                try {

                    faceDetector = TrainHelper.loadClassifierCascade(FaceOperatorRecognition.this, R.raw.frontalface);
                    reloadModel();

                } catch (Exception e) {
                    Log.d(TAG, e.getLocalizedMessage(), e);
                }
                return null;
            }

            @Override
            protected void onPostExecute(Void aVoid) {
                super.onPostExecute(aVoid);
                btn_capture.setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        takePhoto = true;
                    }
                });
                btn_train.setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        train();
                    }
                });
                btn_reset.setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        try {
                            TrainHelper.reset(getBaseContext());
                            trained = false;
                            operators = read();
                            Toast.makeText(getBaseContext(), "Dados reiniciados.", Toast.LENGTH_SHORT).show();
                        } catch (Exception e) {
                            Log.d(TAG, e.getLocalizedMessage(), e);
                        }
                    }
                });
            }
        }.execute();
    }

    //Executa o treinamento das faces
    public void train() {

        new AsyncTask<Void, Void, Void>() {

            @Override
            protected Void doInBackground(Void... voids) {
                try {
                    TrainHelper.train(getBaseContext());

                } catch (Exception e) {
                    Log.d(TAG, e.getLocalizedMessage(), e);
                }
                return null;
            }

            @Override
            protected void onPostExecute(Void aVoid) {
                super.onPostExecute(aVoid);
                try {

                    reloadModel();
                    Toast.makeText(getBaseContext(), "Modelo treinado", Toast.LENGTH_SHORT).show();
                    Toast.makeText(getBaseContext(), "Modelo recarregado", Toast.LENGTH_SHORT).show();

                } catch (Exception e) {
                    Log.d(TAG, e.getLocalizedMessage(), e);
                }
            }


        }.execute();
    }

    //Recalcula a escala da imagem de acordo com a resolução
    @Override
    public void onCameraViewStarted(int width, int height) {
        absoluteFaceSize = (int) (width * 0.15f);
    }

    @Override
    public void onCameraViewStopped() {
        //cameraView.releaseCamera();
    }

    //Captura a imagem para treinamento
    private void capturePhoto(Mat rgbaMat, int id, Integer code_operator, String name_operator) {
        try {
            int i = 1;
            while (i <= 25) {
                TrainHelper.takePhoto(getBaseContext(), id, code_operator, name_operator, i, rgbaMat.clone(), faceDetector);
                Thread.sleep(10);
                i++;
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        takePhoto = false;
    }



    //Realiza o reconhecimento do operador
    private void recognize(opencv_core.Rect dadosFace, Mat grayMat, Mat rgbaMat) {

        Mat detectedFace = new Mat(grayMat, dadosFace);
        resize(detectedFace, detectedFace, new Size(TrainHelper.RESIZE_IMG, TrainHelper.RESIZE_IMG));

        IntPointer label = new IntPointer(1);
        DoublePointer reliability = new DoublePointer(1);

        //faceRecognizer.predict(detectedFace, label, reliability);
        lbphRecognizer.predict(detectedFace, label, reliability);

        int prediction = label.get(0);
        double acceptanceLevel = reliability.get(0);

        String name;


        if (prediction != -1 && acceptanceLevel >= TrainHelper.TOLERANCE_LEVEL) {
            name = "Nao identificado - " + prediction + "-" + acceptanceLevel;
        } else {
            name = operators[prediction] + " - " + prediction + "-" + acceptanceLevel;
        }

        int x = Math.max(dadosFace.tl().x() - 10, 0);
        int y = Math.max(dadosFace.tl().y() - 10, 0);

        putText(rgbaMat, name, new Point(x, y), FONT_HERSHEY_PLAIN, 1, new opencv_core.Scalar(0, 255, 255, 0));
    }

    //Desenho do retangulo na face
    void showDetectedFace(RectVector faces, Mat rgbaMat) {

        int x = faces.get(0).x();
        int y = faces.get(0).y();
        int w = faces.get(0).width();
        int h = faces.get(0).height();

        rectangle(rgbaMat, new Point(x, y), new Point(x + w, y + h), opencv_core.Scalar.BLACK, 1, LINE_8, 0);
    }

    //Label de quando o operador nao esta cadastrado
    void noTrainedLabel(opencv_core.Rect face, Mat rgbaMat) {
        int x = Math.max(face.tl().x() - 10, 0);
        int y = Math.max(face.tl().y() - 10, 0);
        putText(rgbaMat, "Nao cadastrado", new Point(x, y), FONT_HERSHEY_PLAIN, 1, new opencv_core.Scalar(0, 255, 255, 0));
    }

    //Realiza a leitura da imagem e aguarda receber comando para cadastro
    @Override
    public Mat onCameraFrame(Mat rgbaMat) {

        if (faceDetector != null) {

            Mat greyMat = new Mat(rgbaMat.rows(), rgbaMat.cols());
            cvtColor(rgbaMat, greyMat, CV_BGR2GRAY);
            RectVector faces = new RectVector();

            faceDetector.detectMultiScale(greyMat, faces, 1.25f, 3, 1,
                    new Size(absoluteFaceSize, absoluteFaceSize),
                    new Size(4 * absoluteFaceSize, 4 * absoluteFaceSize));

            if (faces.size() == 1) {

                showDetectedFace(faces, rgbaMat);


                if (takePhoto) {

                    int id = operators.length - 1;
                    //alterar para pegar dados da comunicação
                    Operator operator = new Operator(myCodes[id], myOperators[id]);
                    String cadastro = operator.getCode() + "-" + operator.getNome();
                    write(cadastro);
                    capturePhoto(rgbaMat, id, operator.getCode(), operator.getNome());
                    train();

                }
                if (trained) {
                    recognize(faces.get(0), greyMat, rgbaMat);
                } else {
                    noTrainedLabel(faces.get(0), rgbaMat);
                }
            }
            greyMat.release();
        }
        return rgbaMat;
    }

    //Realiza a leitura dos operadores cadastrados
    private String[] read() {

        String dados = "";

        try {

            File fileOperator = new File(getFilesDir(), "dataset");
            File txtOperator = new File(getFilesDir(), DATASET);

            if (!fileOperator.exists()) {
                fileOperator.mkdirs();
            }

            txtOperator.createNewFile();

            InputStream inputStream = new FileInputStream(getFilesDir() + DATASET);
            int fileLen = inputStream.available();
            byte[] fileBuffer = new byte[fileLen];
            inputStream.read(fileBuffer);
            inputStream.close();
            dados = new String(fileBuffer);

            return dados.split("\\,", -1);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }

        return null;

    }

    //Adiciona um novo operador
    private void write(String data) {

        try {

            String caminho = getFilesDir() + DATASET;
            BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(caminho, true));
            bufferedWriter.write(data + ",");
            bufferedWriter.close();

        } catch (IOException e) {
            e.printStackTrace();

        }

    }


    private void reloadModel() {

        if (TrainHelper.isTrained(getBaseContext())) {

            File folder = new File(getFilesDir(), TrainHelper.TRAIN_OPERATORS);
            File f = new File(folder, TrainHelper.EIGEN_CLASSIFIER);
            File l = new File(folder, TrainHelper.LBPH_CLASSIFIER);
            faceRecognizer.read(f.getAbsolutePath());
            lbphRecognizer.read(l.getAbsolutePath());
            trained = true;

        }

        operators = read();

    }

}
