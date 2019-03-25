import socket, select, pickle


class RemoteController:
    '''Implements remote controller of running python program through sockets. 
    That program should accept commands with RemoteReceiver class.
    Commands can be sent as any python serializable objects.'''
    
    
    def __init__(self, port=9000, max_buffer=4096, time_out=4):
        self.port = port
        self.time_out = time_out
        self.max_buffer = max_buffer
        
    def send_commands(self,commands):
        sock = socket.socket()
        sock.settimeout(self.time_out)
        try:
            sock.connect(('localhost', self.port))        
            s = pickle.dumps(commands)
            sock.send(s)
            data = sock.recv(self.max_buffer)
        finally:
            sock.close()
        
        assert data==b'Done!','Command not executed!'



class RemoteReceiver:
    '''Implements remote receiving of commands from remote program.
    That program should send commands with RemoteController class.
    Commands can be sent as any python serializable objects.'''
    
    def __init__(self,port=9090,max_buffer=4096):
        self.sock = socket.socket()
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)   
        self.sock.setblocking(0)
        self.sock.bind(('', port))
        self.sock.listen(1)
        
        self.max_buffer = max_buffer
        
    @property
    def has_commands(self):
        readable_sockets,writable_sockets,_ = select.select([self.sock],[],[],0)
        return (self.sock in readable_sockets)

    def clear(self):
        if self.has_commands:
            self.get_commands()        
        
    def get_commands(self):
        conn, addr = self.sock.accept()
        data = conn.recv(self.max_buffer)
        commands = None
        try:
            commands = pickle.loads(data)
            conn.send(b'Done!')
        finally:
            conn.close()
        return commands
            
    def __del__(self):
        self.sock.close()    


'''
CONTROLLER:
RemoteController(port=9000).send_commands({'learning_rate':0.0001})

RECEIVER:
receiver = RemoteReceiver(port=9000)	
receiver.clear()
while True:
	if receiver.has_commands:
		commands = receiver.get_commands()
		if 'stop' in commands:
			break
			
		if 'learning_rate' in commands:
			learning_rate = commands['learning_rate']
receiver.sock.close()			
'''
