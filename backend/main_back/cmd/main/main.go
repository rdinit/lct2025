package main

import (
	"fmt"
	"log"
	"main_back/internal/datamanagers"
	"math/rand"
	"net/http"

	"github.com/gorilla/websocket"
)

var rn int

var dataManagers = make(map[string]*datamanagers.DataManager)

var dataMerger = datamanagers.NewDataMerger()

var upgrader = websocket.Upgrader{CheckOrigin: func(r *http.Request) bool {
	return true
}}

func health_handler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprint(w, "healthy ", rn)
}

func new_data_handler(w http.ResponseWriter, r *http.Request) {

	sensor_id := r.URL.Query().Get("sensor_id")
	fmt.Println(sensor_id)

	c, err := upgrader.Upgrade(w, r, nil)

	if err != nil {
		log.Println("upgrade error:", err)
		return
	}
	dataManager, ok := dataManagers[sensor_id]
	if !ok {
		log.Println("sensor id not found:", sensor_id)
		return
	}

	c.SetCloseHandler(func(code int, text string) error {
		return nil
	})

	go func() {
		for {
			mt, message, err := c.ReadMessage()
			if err != nil {
				if _, ok := err.(*websocket.CloseError); ok {
					log.Println("Stop recieving. Disconnect")

				} else {
					log.Println("reading error :", err)
				}
				break
			}
			if mt != websocket.TextMessage {
				continue
			}

			dataManager.ProcessMessage(message)
		}
	}()
}

func new_reader_handler(w http.ResponseWriter, r *http.Request) {
	sensor_id := r.URL.Query().Get("sensor_id")

	c, err := upgrader.Upgrade(w, r, nil)

	if err != nil {
		log.Println("upgrade error:", err)
		return
	}

	dataManager, ok := dataManagers[sensor_id]
	if !ok {
		log.Println("sensor id not found:", sensor_id)
		return
	}
	c.SetCloseHandler(func(code int, text string) error {
		dataManager.RemoveOutputConnection(c)
		log.Println("disconnected:")
		return nil
	})

	dataManager.AddOutputConnection(c)

}

func get_forecast_handler(w http.ResponseWriter, r *http.Request) {
	c, err := upgrader.Upgrade(w, r, nil)

	if err != nil {
		log.Println("upgrade error:", err)
		return
	}

	c.SetCloseHandler(func(code int, text string) error {
		dataMerger.Ml_handlers["forecast"].RemoveOutputConnection(c)
		log.Println("disconnected:")
		return nil
	})

	dataMerger.Ml_handlers["forecast"].AddOutputConnection(c)

}

func get_anomaly_handler(w http.ResponseWriter, r *http.Request) {
	c, err := upgrader.Upgrade(w, r, nil)

	if err != nil {
		log.Println("upgrade error:", err)
		return
	}

	c.SetCloseHandler(func(code int, text string) error {
		dataMerger.Ml_handlers["anomaly"].RemoveOutputConnection(c)
		log.Println("disconnected:")
		return nil
	})

	dataMerger.Ml_handlers["anomaly"].AddOutputConnection(c)
}
func get_classify_handler(w http.ResponseWriter, r *http.Request) {
	c, err := upgrader.Upgrade(w, r, nil)

	if err != nil {
		log.Println("upgrade error:", err)
		return
	}

	c.SetCloseHandler(func(code int, text string) error {
		dataMerger.Ml_handlers["classify"].RemoveOutputConnection(c)
		log.Println("disconnected:")
		return nil
	})

	dataMerger.Ml_handlers["classify"].AddOutputConnection(c)
}

func make_forecast_handler(w http.ResponseWriter, r *http.Request) {
	c, err := upgrader.Upgrade(w, r, nil)

	if err != nil {
		log.Println("upgrade error:", err)
		return
	}

	c.SetCloseHandler(func(code int, text string) error {
		dataMerger.Ml_handlers["forecast"].DisconnectML()
		log.Println("disconnected:")
		return nil
	})
	dataMerger.Ml_handlers["forecast"].ConnectML(c)
	go func() {
		for {
			mt, message, err := c.ReadMessage()
			if err != nil {
				break
			}
			if mt != websocket.TextMessage {
				log.Println("no text message, closing connection")
				//log.Println("message type: ", mt, "message: ", message, "error: ", err)
				continue
			}
			//log.Println("recieved message: ", string(message))
			dataMerger.Ml_handlers["forecast"].SendUpdate(message)
		}
	}()
}

func make_anomaly_handler(w http.ResponseWriter, r *http.Request) {
	c, err := upgrader.Upgrade(w, r, nil)

	if err != nil {
		log.Println("upgrade error:", err)
		return
	}

	c.SetCloseHandler(func(code int, text string) error {
		dataMerger.Ml_handlers["anomaly"].DisconnectML()
		log.Println("disconnected:")
		return nil
	})
	dataMerger.Ml_handlers["anomaly"].ConnectML(c)
	go func() {
		for {
			mt, message, err := c.ReadMessage()
			if err != nil {
				break
			}
			if mt != websocket.TextMessage {
				continue
			}
			//log.Println("recieved message: ", string(message))
			dataMerger.Ml_handlers["anomaly"].SendUpdate(message)
		}
	}()
}

func make_classify_handler(w http.ResponseWriter, r *http.Request) {
	c, err := upgrader.Upgrade(w, r, nil)

	if err != nil {
		log.Println("upgrade error:", err)
		return
	}

	c.SetCloseHandler(func(code int, text string) error {
		dataMerger.Ml_handlers["classify"].DisconnectML()
		log.Println("disconnected:")
		return nil
	})
	dataMerger.Ml_handlers["classify"].ConnectML(c)
	go func() {
		for {
			mt, message, err := c.ReadMessage()
			if err != nil {
				if _, ok := err.(*websocket.CloseError); ok {
					log.Println("Stop recieving. Disconnect")

				} else {
					log.Println("reading error :", err)
				}
				break
			}
			if mt != websocket.TextMessage {
				log.Println("no text message, closing connection")
				//log.Println("message type: ", mt, "message: ", message, "error: ", err)
				continue
			}
			//log.Println("recieved message: ", string(message))
			dataMerger.Ml_handlers["classify"].SendUpdate(message)
		}
	}()
}

func main() {
	rn = rand.Intn(100)
	fmt.Println(rn)

	dataManagers["bpm"] = datamanagers.NewDataManager("bpm", dataMerger)
	dataManagers["uterus"] = datamanagers.NewDataManager("uterus", dataMerger)

	dataMerger.Ml_handlers["forecast"] = datamanagers.NewMLHandler(0, 3000)
	dataMerger.Ml_handlers["anomaly"] = datamanagers.NewMLHandler(0, 1)
	dataMerger.Ml_handlers["classify"] = datamanagers.NewMLHandler(250, 1000)

	http.HandleFunc("/health", health_handler)
	http.HandleFunc("/send", new_data_handler)
	http.HandleFunc("/data", new_reader_handler)

	http.HandleFunc("/get_forecast", get_forecast_handler)
	http.HandleFunc("/make_forecast", make_forecast_handler)
	http.HandleFunc("/get_anomaly", get_anomaly_handler)
	http.HandleFunc("/make_anomaly", make_anomaly_handler)
	http.HandleFunc("/get_classify", get_classify_handler)
	http.HandleFunc("/make_classify", make_classify_handler)

	http.HandleFunc("/datasender", func(w http.ResponseWriter, r *http.Request) {
		http.ServeFile(w, r, "datasender.html")
	})
	http.HandleFunc("/plot", func(w http.ResponseWriter, r *http.Request) {
		http.ServeFile(w, r, "plot.html")
	})

	http.ListenAndServe(":8082", nil)
}
