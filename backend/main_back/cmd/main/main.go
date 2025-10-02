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

func back2front_handler(w http.ResponseWriter, r *http.Request) {
	model := r.URL.Query().Get("model")

	c, err := upgrader.Upgrade(w, r, nil)

	if err != nil {
		log.Println("upgrade error:", err)
		return
	}

	_, ok := dataMerger.Ml_handlers[model]
	if !ok {
		log.Println("model not found:", model)
		return
	}

	c.SetCloseHandler(func(code int, text string) error {
		dataMerger.Ml_handlers[model].RemoveOutputConnection(c)
		log.Println("disconnected:")
		return nil
	})

	dataMerger.Ml_handlers[model].AddOutputConnection(c)

}

func back2ml_handler(w http.ResponseWriter, r *http.Request) {
	model := r.URL.Query().Get("model")

	c, err := upgrader.Upgrade(w, r, nil)

	if err != nil {
		log.Println("upgrade error:", err)
		return
	}

	_, ok := dataMerger.Ml_handlers[model]
	if !ok {
		log.Println("model not found:", model)
		return
	}
	c.SetCloseHandler(func(code int, text string) error {
		dataMerger.Ml_handlers[model].DisconnectML()
		log.Println("disconnected:")
		return nil
	})
	dataMerger.Ml_handlers[model].ConnectML(c)
	go func() {
		for {
			mt, message, err := c.ReadMessage()
			if err != nil {
				break
			}
			if mt != websocket.TextMessage {
				continue
			}
			dataMerger.Ml_handlers[model].SendUpdate(message)
		}
	}()
}

func main() {
	rn = rand.Intn(100)
	fmt.Println(rn)

	dataManagers["bpm"] = datamanagers.NewDataManager("bpm", dataMerger)
	dataManagers["uterus"] = datamanagers.NewDataManager("uterus", dataMerger)

	dataMerger.Ml_handlers["forecast"] = datamanagers.NewMLHandler(0, 80)
	dataMerger.Ml_handlers["anomaly"] = datamanagers.NewMLHandler(0, 200)
	dataMerger.Ml_handlers["classify"] = datamanagers.NewMLHandler(750, 1000)
	dataMerger.Ml_handlers["diagnoses"] = datamanagers.NewMLHandler(500, 1000)

	http.HandleFunc("/health", health_handler)
	http.HandleFunc("/send", new_data_handler)
	http.HandleFunc("/data", new_reader_handler)

	http.HandleFunc("/back2front", back2front_handler)
	http.HandleFunc("/back2ml", back2ml_handler)

	http.HandleFunc("/datasender", func(w http.ResponseWriter, r *http.Request) {
		http.ServeFile(w, r, "datasender.html")
	})
	http.HandleFunc("/plot", func(w http.ResponseWriter, r *http.Request) {
		http.ServeFile(w, r, "plot.html")
	})

	http.ListenAndServe(":8082", nil)
}
