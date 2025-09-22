package moe.kotorinminami.sensor;
interface IGetEventService {
    void destroy() = 16777114;
    void getEvent(int port) = 2;
    String getActivity() = 3;
}