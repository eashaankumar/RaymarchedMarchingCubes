using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

[RequireComponent(typeof(TMPro.TMP_Text))]
public class FPSRenderer : MonoBehaviour
{
    TMPro.TMP_Text text;
    int frames;
    float time;

    // Start is called before the first frame update
    void Start()
    {
        text = GetComponent<TMPro.TMP_Text>();
    }

    // Update is called once per frame
    void Update()
    {
        if (time >= 1)
        {
            text.text = frames + "";
            frames = 0;
            time = 0;
        }
        time += Time.deltaTime;
        frames++;
    }
}
