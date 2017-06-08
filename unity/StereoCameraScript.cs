using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class StereoCameraScript : MonoBehaviour {

    public double cameraAzimuthStep = 5.0;
    public double cameraElevationMin = 0.0;
    public double cameraElevationMax = 30.0;
    public double cameraElevationStep = 10.0;
    public double cameraRadius = 1.0;
    public double cameraFOV = 60.0;
    public int imgHeight = 512;
    public int imgWidth = 512;

    private bool captureImages = false;
    private string savePath = "default";
    private List<List<Camera>> cameras = new List<List<Camera>>();

	// Use this for initialization
	void Start () {
        int i, j;
        int numberOfCameras = (int)System.Math.Floor(360.0 / cameraAzimuthStep);
        int numberOfLayers = (int)System.Math.Ceiling((cameraElevationMax - cameraElevationMin) / cameraElevationStep) + 1;
        double actualCameraAzimuthStep = 2 * System.Math.PI / numberOfCameras;
        double actualCameraElevationMin = System.Math.PI * cameraElevationMin / 180.0;
        double actualCameraElevationStep = System.Math.PI * cameraElevationStep / 180.0;
        for (i = 0; i < numberOfLayers; i++)
        {
            cameras.Add(new List<Camera>());
            for (j = 0; j < numberOfCameras; j++)
            {
                GameObject obj = new GameObject();
                Camera camera = obj.AddComponent<Camera>();
                camera.clearFlags = CameraClearFlags.SolidColor;
                camera.backgroundColor = Color.white;
                camera.fieldOfView = (float)cameraFOV;
                camera.depthTextureMode = DepthTextureMode.Depth;
                float theta = (float)(j * actualCameraAzimuthStep);
                float phi = (float)(i * actualCameraElevationStep + actualCameraElevationMin);
                Vector3 position = new Vector3();
                position.x = (float)(cameraRadius * System.Math.Cos(phi) * System.Math.Cos(theta));
                position.z = (float)(cameraRadius * System.Math.Cos(phi) * System.Math.Sin(theta));
                position.y = (float)(cameraRadius * System.Math.Sin(phi));
                phi *= (float)(180.0 / System.Math.PI);
                theta *= (float)(180.0 / System.Math.PI);
                Quaternion rotation = Quaternion.Euler(phi, -90.0f-theta, 0);
                obj.transform.position = position;
                obj.transform.rotation = rotation;
                cameras[i].Add(camera);
            }
        }  
	}
	
	// Update is called once per frame
	void LateUpdate () {
        captureImages |= Input.GetKeyUp("c");
        if (captureImages)
        {
            int i, j;
            int numberOfCameras = (int)System.Math.Floor(360.0 / cameraAzimuthStep);
            int numberOfLayers = (int)System.Math.Ceiling((cameraElevationMax - cameraElevationMin) / cameraElevationStep) + 1;

            for (i = 0; i < numberOfLayers; i++)
            {
                int elevation = (int)(cameraElevationMin + i * cameraElevationStep);
                for (j = 0; j < numberOfCameras; j++)
                {
                    int azimuth = (int)(j * 360.0 / numberOfCameras);
                    Camera camera = cameras[i][j];

                    RenderTexture rtImg = new RenderTexture(imgWidth, imgHeight, 24);
                    camera.targetTexture = rtImg;
                    Texture2D captureImg = new Texture2D(imgWidth, imgHeight, TextureFormat.RGB24, false);
                    camera.Render();
                    RenderTexture.active = rtImg;
                    captureImg.ReadPixels(new Rect(0, 0, imgWidth, imgHeight), 0, 0);
                    camera.targetTexture = null;
                    RenderTexture.active = null;
                    Destroy(captureImg);
                    Destroy(rtImg);
                    byte[] img = captureImg.EncodeToPNG();

                    RenderTexture rtDepth = new RenderTexture(imgWidth, imgHeight, 24);
                    camera.gameObject.AddComponent<RenderDepthScript>();
                    camera.targetTexture = rtDepth;
                    Texture2D captureDepth = new Texture2D(imgWidth, imgHeight, TextureFormat.RGB24, false);
                    camera.Render();
                    RenderTexture.active = rtDepth;
                    captureDepth.ReadPixels(new Rect(0, 0, imgWidth, imgHeight), 0, 0);
                    camera.targetTexture = null;
                    RenderTexture.active = null;
                    Destroy(captureDepth);
                    Destroy(rtDepth);
                    Destroy(camera.gameObject.GetComponent<RenderDepthScript>());
                    byte[] depth = captureDepth.EncodeToPNG();

                    string imgName = string.Format("{2}/img_{0}_{1}.png", elevation, azimuth, savePath);
                    string depthName = string.Format("{2}/depth_{0}_{1}.png", elevation, azimuth, savePath);
                    System.IO.File.WriteAllBytes(imgName, img);
                    System.IO.File.WriteAllBytes(depthName, depth);
                }
            }

            captureImages = false;
        }
	}

    public void CaptureImages(string savePath)
    {
        this.savePath = savePath;
        captureImages = true;
    }
}
