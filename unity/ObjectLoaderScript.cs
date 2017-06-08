using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ObjectLoaderScript : MonoBehaviour {

    public StereoCameraScript script;

    private bool done = false;
    private int frame = 0;
    private int catIndex = 0;
    private int objIndex = 0;
    private GameObject go;

	// Use this for initialization
	void Start () {
	}
	
	// Update is called once per frame
	void Update () {
        if (!done)
        {
            string[] categories = System.IO.Directory.GetDirectories("Assets\\Resources\\ShapeNetCore");
            string cat = categories[catIndex];
            string[] objects = System.IO.Directory.GetDirectories(cat);
            string obj = objects[objIndex];
            string[] path = obj.Split('\\');

            if (frame == 0)
            {
                if (go != null)
                    Destroy(go);
                System.GC.Collect();
            }
            else if (frame == 1)
            {
                string assetName = string.Format("ShapeNetCore/{0}/{1}/models/model_normalized", path[3], path[4]);
                Debug.Log(assetName);

                go = Instantiate(Resources.Load(assetName, typeof(GameObject)), Vector3.zero, Quaternion.identity) as GameObject;
            }
            else if (frame == 2)
            {
                if (!System.IO.Directory.Exists(path[3]))
                    System.IO.Directory.CreateDirectory(path[3]);
                if (!System.IO.Directory.Exists(string.Format("{0}\\{1}", path[3], path[4])))
                    System.IO.Directory.CreateDirectory(string.Format("{0}\\{1}", path[3], path[4]));

                script.CaptureImages(string.Format("{0}\\{1}", path[3], path[4]));
                objIndex += 1;
                if (objIndex == objects.Length)
                {
                    objIndex = 0;
                    catIndex += 1;
                    if (catIndex == categories.Length)
                        done = true;
                }
            }
            frame = (frame + 1) % 3;
        }
	}
}
