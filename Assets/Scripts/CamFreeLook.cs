using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(Camera))]
public class CamFreeLook : MonoBehaviour
{
    [SerializeField]
    float lookSensitivity;
    [SerializeField]
    float moveSpeed;

    Vector3 velocity;
    float yaw, pitch;
    Vector2 mousePos;
    Vector2 mouseDelta;

    // Start is called before the first frame update
    void Start()
    {
        Application.targetFrameRate = 400;
    }

    // Update is called once per frame
    void Update()
    {
        // Look
        Cursor.lockState = CursorLockMode.Locked;
        mouseDelta = new Vector2(Input.GetAxis("Mouse X"), Input.GetAxis("Mouse Y"));
        mouseDelta *= lookSensitivity;
        pitch += mouseDelta.y;
        yaw += mouseDelta.x;
        mousePos = Input.mousePosition;
        transform.rotation = Quaternion.Euler(-pitch, yaw, 0);

        // Move
        velocity = transform.forward * Input.GetAxis("Vertical") + transform.right * Input.GetAxis("Horizontal");
        velocity *= moveSpeed;
        //transform.Translate(velocity * Time.deltaTime);
        transform.position += transform.forward * moveSpeed * Time.deltaTime * Input.GetAxis("Vertical") + 
                               transform.right * moveSpeed * Time.deltaTime * Input.GetAxis("Horizontal");
    }
}
