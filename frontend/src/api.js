import ky from 'ky'
const api = ky.create({ prefixUrl: 'http://localhost:8000', timeout: 600000 })
export default api
